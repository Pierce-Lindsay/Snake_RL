"""! @file mlp.py
@brief Deep Q-Network (DQN) agent for Snake using a multilayer perceptron.

The agent observes an 11-feature vector (danger, heading, food direction), or with
--extended-state 14 features (adds free space per move, to avoid self-trapping), and an
MLP approximates Q(s, a) for the three relative actions (straight, right, left).
Training uses experience replay and a periodically synced target network.
"""

import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import snake


class MLP(nn.Module):
    """! Q-network following Sebastianelli et al. (2021).

    11 inputs -> 3 x (100 fully connected + ReLU + Dropout) -> 3 outputs + SoftMax.
    Outputs correspond to (nothing, left, right).
    """

    def __init__(self, in_dim=11, hidden=100, out_dim=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def q_values(self, x):
        """! Pre-SoftMax outputs, used as Q-values for training (unbounded)."""
        return self.net(x)

    def forward(self, x):
        """! Action probabilities. argmax equals argmax of q_values."""
        return self.softmax(self.net(x))


class SnakeEnv(snake.Snake):
    """! Snake game wrapped as an RL environment with a feature-vector state."""

    STRAIGHT, LEFT, RIGHT = 0, 1, 2  # matches MLP output order

    def __init__(self, extended=False):
        """! @param extended Add 3 free-space features (14 inputs instead of the paper's 11)."""
        self.extended = extended
        super().__init__()

    @property
    def state_dim(self):
        return 14 if self.extended else 11

    def reset(self):
        super().__init__()
        self.displacement = np.array([0.0, -self.SNAKE_BLOCK])
        self.steps_since_food = 0
        return self.get_state()

    def _turn_right(self):
        return np.array([-self.displacement[1], self.displacement[0]])

    def _turn_left(self):
        return np.array([self.displacement[1], -self.displacement[0]])

    def _is_danger(self, direction):
        p = self.position + direction
        if p[0] < 0 or p[0] >= self.SCREEN_WIDTH or p[1] < 0 or p[1] >= self.SCREEN_HEIGHT:
            return True
        return any(np.array_equal(p, seg) for seg in self.snake_list[:-1])

    def _free_space(self, direction):
        """! Fraction of the snake's length that fits in the region reachable via `direction`.

        Flood fill from the next cell, treating the body (minus the tail, which moves
        away) as walls. Capped at snake_length, so 1.0 means "room for the whole body"
        and anything lower means that move leads into a dead end.
        """
        b = self.SNAKE_BLOCK
        cols, rows = self.SCREEN_WIDTH // b, self.SCREEN_HEIGHT // b
        cap = self.snake_length
        blocked = {(int(x // b), int(y // b)) for x, y in self.snake_list[1:]}
        start = (int((self.position[0] + direction[0]) // b), int((self.position[1] + direction[1]) // b))
        if not (0 <= start[0] < cols and 0 <= start[1] < rows) or start in blocked:
            return 0.0
        seen, stack = {start}, [start]
        while stack and len(seen) < cap:
            x, y = stack.pop()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= n[0] < cols and 0 <= n[1] < rows and n not in blocked and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return min(len(seen), cap) / cap

    def get_state(self):
        dx, dy = self.displacement
        fx, fy = self.food_position - self.position
        features = [
            self._is_danger(self.displacement),
            self._is_danger(self._turn_right()),
            self._is_danger(self._turn_left()),
            dx < 0, dx > 0, dy < 0, dy > 0,   # heading: left, right, up, down
            fx < 0, fx > 0, fy < 0, fy > 0,   # food: left, right, up, down
        ]
        if self.extended:  # free space: straight, right, left
            features += [self._free_space(self.displacement),
                         self._free_space(self._turn_right()),
                         self._free_space(self._turn_left())]
        return np.array(features, dtype=np.float32)

    def step(self, action):
        """! Apply action, advance one tick. @return (state, reward, done)"""
        old_dist = np.linalg.norm(self.food_position - self.position)
        if action == self.RIGHT:
            self.displacement = self._turn_right()
        elif action == self.LEFT:
            self.displacement = self._turn_left()
        self.process()
        self.steps_since_food += 1

        if self.game_over:
            return self.get_state(), -10.0, True
        if self.is_eating:
            self.steps_since_food = 0
            return self.get_state(), 10.0, False
        # Stop endless loops that never reach the food
        if self.steps_since_food > 100 * self.snake_length:
            self.game_over = True
            return self.get_state(), -10.0, True
        new_dist = np.linalg.norm(self.food_position - self.position)
        return self.get_state(), (0.1 if new_dist < old_dist else -0.1), False


class DQNAgent:
    """! Epsilon-greedy DQN agent with replay buffer and target network."""

    def __init__(self, lr=1e-3, gamma=0.9, batch_size=256, buffer_size=100_000,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995, target_sync=500, dropout=0.1,
                 in_dim=11):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = MLP(in_dim=in_dim, dropout=dropout).to(self.device)
        self.target = MLP(in_dim=in_dim, dropout=dropout).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()  # no dropout when computing targets
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.gamma, self.batch_size, self.target_sync = gamma, batch_size, target_sync
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.learn_steps = 0

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(3)
        self.q.eval()  # disable dropout for action selection
        with torch.no_grad():
            s = torch.as_tensor(state, device=self.device).unsqueeze(0)
            return int(self.q(s).argmax(1).item())

    def remember(self, *transition):
        self.memory.append(transition)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = (torch.as_tensor(np.array(x), device=self.device) for x in zip(*batch))
        self.q.train()
        q = self.q.q_values(s).gather(1, a.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r.float() + self.gamma * self.target.q_values(s2).max(1).values * (1 - d.float())
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())


def render(env, surface, font):
    import pygame
    surface.fill((50, 153, 213))
    b = env.SNAKE_BLOCK
    pygame.draw.rect(surface, (213, 50, 80), [*env.food_position, b, b])
    for seg in env.snake_list[:-1]:
        pygame.draw.rect(surface, (0, 0, 0), [*seg, b, b])
    pygame.draw.rect(surface, (255, 255, 255), [*env.snake_list[-1], b, b])
    surface.blit(font.render(f"Score: {env.snake_length - 1}", True, (0, 0, 0)), [0, 0])
    pygame.display.update()
    pygame.event.pump()


def train(episodes, display=False, model_path="mlp_snake.pt", extended=False, **agent_kwargs):
    env = SnakeEnv(extended)
    agent = DQNAgent(in_dim=env.state_dim, **agent_kwargs)
    scores, best = [], -1
    if display:
        import pygame
        pygame.init()
        surface = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        font = pygame.font.SysFont("bahnschrift", 25)

    for ep in range(episodes):
        state, done = env.reset(), False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if display:
                render(env, surface, font)
        agent.eps = max(agent.eps_end, agent.eps * agent.eps_decay)

        score = env.snake_length - 1
        scores.append(score)
        if score > best:
            best = score
            torch.save(agent.q.state_dict(), model_path)
        if ep % 10 == 0:
            print(f"Episode {ep}: score={score} best={best} "
                  f"avg10={np.mean(scores[-10:]):.1f} eps={agent.eps:.3f}")
    return agent, scores


def play(model_path, games=10, display=True):
    """! Run the saved greedy policy and report scores."""
    state_dict = torch.load(model_path, map_location="cpu")
    in_dim = state_dict["net.0.weight"].shape[1]  # 11 = paper state, 14 = extended
    env, agent = SnakeEnv(extended=in_dim == 14), DQNAgent(in_dim=in_dim)
    agent.q.load_state_dict(state_dict)
    if display:
        import pygame
        pygame.init()
        surface = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        font = pygame.font.SysFont("bahnschrift", 25)
        clock = pygame.time.Clock()
    scores = []
    for _ in range(games):
        state, done = env.reset(), False
        while not done:
            state, _, done = env.step(agent.act(state, greedy=True))
            if display:
                render(env, surface, font)
                clock.tick(env.SNAKE_SPEED * 2)
        scores.append(env.snake_length - 1)
    print(f"Scores: {scores}  mean={np.mean(scores):.1f}")
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN (MLP) agent for Snake")
    parser.add_argument("-e", "--episodes", type=int, default=1000)
    parser.add_argument("-d", "--display", action="store_true")
    parser.add_argument("--play", action="store_true", help="watch a saved model instead of training")
    parser.add_argument("-m", "--model", default="mlp_snake.pt")
    parser.add_argument("--eval", type=int, default=0, metavar="N",
                        help="after training, score the saved model on N headless greedy games")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eps-decay", type=float, default=0.995)
    parser.add_argument("--target-sync", type=int, default=500)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--extended-state", action="store_true",
                        help="add 3 free-space (flood fill) inputs: 14 inputs instead of 11")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.play:
        play(args.model)
    else:
        _, scores = train(args.episodes, args.display, args.model, args.extended_state,
                          lr=args.lr, gamma=args.gamma,
                          batch_size=args.batch_size, eps_decay=args.eps_decay,
                          target_sync=args.target_sync, dropout=args.dropout)
        if args.eval:
            ev = play(args.model, games=args.eval, display=False)
            print(f"EVAL games={args.eval} mean={np.mean(ev):.2f} max={max(ev)} "
                  f"train_last100={np.mean(scores[-100:]):.2f}")
        try:
            import matplotlib.pyplot as plt
            w = max(2, min(100, len(scores) // 10))
            plt.figure(figsize=(10, 6))
            plt.plot(scores, alpha=0.3, label="Score")
            plt.plot(range(w - 1, len(scores)), np.convolve(scores, np.ones(w) / w, "valid"),
                     color="red", label=f"Moving average ({w})")
            plt.xlabel("Episode"); plt.ylabel("Score"); plt.title("MLP DQN training")
            plt.legend(); plt.grid(alpha=0.3)
            plt.savefig("training_progress_mlp.png")
        except ImportError:
            print("matplotlib not installed - skipping progress plot")
