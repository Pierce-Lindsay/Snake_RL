"""! @file snake.py
@brief Implementation of the core Snake game mechanics.

This module provides the base Snake game implementation, handling core game mechanics
such as movement, collision detection, and food placement. It serves as the foundation
for the Snake AI implementation.

@author Tristan Mallet
@date November 2024
"""

import random
import numpy as np


class Snake:
    """! @brief Base class implementing core Snake game mechanics.
    
    This class manages fundamental Snake game operations including:
    - Snake movement and growth
    - Food placement and consumption
    - Collision detection with walls and self
    - Game state management
    
    The game operates on a grid-based system where both the snake and food
    align to fixed block positions.
    """

    # Game configuration constants
    SCREEN_WIDTH = 400   #!< Width of game screen in pixels
    SCREEN_HEIGHT = 300  #!< Height of game screen in pixels
    SNAKE_BLOCK = 10     #!< Size of each snake/food block in pixels
    SNAKE_SPEED = 15     #!< Game update frequency (moves per second)

    def __init__(self, numbered_fruits=False, fruit_count=5):
        """! Initialize a new Snake game instance.
        
        Sets up the initial game state with:
        - Snake positioned at screen center
        - Initial snake length of 1
        - Random food placement
        - No initial movement
        """
        # Game state flags
        self.game_over = False  #!< Tracks if game has ended due to collision
        self.is_eating = False  #!< Tracks if snake is currently eating food
        self.game_started = False
        
        # Snake properties
        self.snake_list = []    #!< List of coordinates for each snake segment
        self.snake_length = 1   #!< Current length of the snake
        
        # Position and movement
        self.position = np.array([
            self.SCREEN_WIDTH / 2,   # Center x-coordinate
            self.SCREEN_HEIGHT / 2    # Center y-coordinate
        ])  #!< Current position of snake head
        self.displacement = np.zeros(2,)  #!< Current movement vector
        
        self._validate_fruit_count(fruit_count)
        self.numbered_fruits = numbered_fruits
        self.fruit_count = fruit_count
        self.create_food()

    def update_snake_position(self):
        """! Update snake head position based on current displacement.
        
        Adds the current displacement vector to the snake's head position,
        effectively moving the snake one step in its current direction.
        This is called each game tick before collision detection.
        """
        self.position += self.displacement

    def check_collisions(self):
        """! Check for game-ending collisions.
        
        Detects collisions with:
        1. Wall collisions: Snake head touches screen boundaries
        2. Self collisions: Snake head touches any part of its body
        3. Fruits numbered 2-5, which act as obstacles
        
        Sets game_over flag to True if any collision is detected.
        """
        # Check wall collisions
        if (self.position[0] >= self.SCREEN_WIDTH
            or self.position[0] < 0
            or self.position[1] >= self.SCREEN_HEIGHT
            or self.position[1] < 0):
            self.game_over = True
            print("Game over: Screen border")

        # Check self collisions
        for segment in self.snake_list[:-1]:  # Exclude head from check
            if np.array_equal(segment, self.position):
                self.game_over = True
                print("Game over: Own body crash")

        if self.numbered_fruits:
            for fruit_position in self.fruit_positions[1:]:
                if np.array_equal(self.position, fruit_position):
                    self.game_over = True
                    print("Game over: Hit a fruit numbered above 1")
                    break

    def update_snake_list(self):
        """! Update the list of snake body segments.
        
        Maintains the snake's body by:
        1. Adding current head position to body list
        2. Removing oldest segment if exceeding current length
        
        This creates the illusion of movement while maintaining proper length.
        """
        self.snake_list.append([self.position[0], self.position[1]])
        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]

    def create_food(self):
        """! Generate classic food or the selected number of numbered fruits.
        
        Places food randomly on screen with constraints:
        - Aligns to SNAKE_BLOCK grid
        - Keeps full blocks within screen boundaries
        - Avoids the snake and other fruits
        """
        if not self.numbered_fruits:
            self.food_position = np.array((
                round(random.randrange(
                    0, self.SCREEN_WIDTH - self.SNAKE_BLOCK) / 10.0) * 10.0,
                round(random.randrange(
                    0, self.SCREEN_HEIGHT - self.SNAKE_BLOCK) / 10.0) * 10.0
            ))
            self.fruit_positions = [self.food_position]
            return

        self.fruit_positions = []
        for _ in range(self.fruit_count):
            position = self._random_empty_position()
            if position is None:
                self.game_over = True
                break
            self.fruit_positions.append(position)

        self.food_position = self.fruit_positions[0] if self.fruit_positions else None

    @staticmethod
    def _validate_fruit_count(fruit_count):
        if fruit_count not in (3, 4, 5):
            raise ValueError("fruit_count must be 3, 4, or 5")

    def set_numbered_fruits(self, enabled, fruit_count=None):
        """! Set numbered-fruit mode and optionally select its fruit count."""
        if fruit_count is not None:
            self._validate_fruit_count(fruit_count)
        if self.game_started:
            return
        if fruit_count is not None:
            self.fruit_count = fruit_count
        self.numbered_fruits = bool(enabled)
        self.create_food()

    def toggle_numbered_fruits(self):
        """! Toggle numbered-fruit mode and return the new state."""
        if self.game_started:
            return self.numbered_fruits
        self.set_numbered_fruits(not self.numbered_fruits)
        return self.numbered_fruits

    def _random_empty_position(self):
        """! Return a random grid position not occupied by the snake or a fruit."""
        occupied = {
            tuple(segment) for segment in self.snake_list
        }
        occupied.add(tuple(self.position))
        occupied.update(tuple(position) for position in self.fruit_positions)

        available = [
            (x, y)
            for x in range(0, self.SCREEN_WIDTH, self.SNAKE_BLOCK)
            for y in range(0, self.SCREEN_HEIGHT, self.SNAKE_BLOCK)
            if (x, y) not in occupied
        ]
        if not available:
            return None

        return np.array(random.choice(available), dtype=float)

    def eat_food(self):
        """! Check for food consumption and handle growth.
        
        Performs these operations:
        1. Checks if snake head overlaps with fruit 1
        2. Increases snake length if fruit 1 is eaten
        3. Shifts remaining fruits down and generates a new fruit at the selected count
        4. Updates is_eating flag for reward calculation
        """
        if self.game_over:
            self.is_eating = False
            return

        self.is_eating = np.array_equal(self.position, self.food_position)
        if self.is_eating:
            self.snake_length += 1
            if self.numbered_fruits:
                self.fruit_positions = self.fruit_positions[1:]
                new_fruit = self._random_empty_position()
                if new_fruit is None:
                    self.game_over = True
                    print("Game over: No space for a new fruit")
                else:
                    self.fruit_positions.append(new_fruit)
                self.food_position = self.fruit_positions[0]
            else:
                self.create_food()

    def process(self):
        """! Process one complete game tick.
        
        Executes main game loop operations in order:
        1. Updates snake head position
        2. Checks for collisions
        3. Updates snake body segments
        4. Handles food consumption
        
        This method should be called once per game tick/frame.
        """
        self.game_started = True
        self.update_snake_position()  # Move snake head
        self.check_collisions()       # Check for game-ending conditions
        self.update_snake_list()      # Update body positions
        self.eat_food()               # Handle food interactions
