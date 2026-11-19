import snake
import numpy as np

COLLISION_FORGIVENESS = 0.25
def check_wall_collision(new_pos: np.ndarray):
    collisions = [False, False]
    if  new_pos[0] >= snake.Snake.SCREEN_WIDTH - snake.Snake.SNAKE_BLOCK or new_pos[0] < 0:
        collisions[0] = True
    if  new_pos[1] >= snake.Snake.SCREEN_HEIGHT - snake.Snake.SNAKE_BLOCK or new_pos[1] < 0:
        collisions[1] = True
    return collisions
 #returns the index of the snake that is collided with or None if not collided
def check_snake_collision(new_pos: np.ndarray, snake_body):
    for i in range(len(snake_body)):
        if np.allclose(snake_body[i], new_pos, atol = snake.Snake.SNAKE_BLOCK* COLLISION_FORGIVENESS ):
            return i
    return None