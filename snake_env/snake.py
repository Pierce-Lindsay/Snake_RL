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

    def __init__(self):
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
        
        # Snake properties
        self.snake_list = []    #!< List of coordinates for each snake segment
        self.snake_length = 1   #!< Current length of the snake
        self.walls = []
        self.max_walls = 3
        
        # Position and movement
        self.position = np.array([
            self.SCREEN_WIDTH / 2,   # Center x-coordinate
            self.SCREEN_HEIGHT / 2    # Center y-coordinate
        ])  #!< Current position of snake head
        self.displacement = np.zeros(2,)  #!< Current movement vector
        
        # Initialize food
        self.create_food()  # Place first food item

    def update_snake_position(self):
        """! Update snake head position based on current displacement.
        
        Adds the current displacement vector to the snake's head position,
        effectively moving the snake one step in its current direction.
        This is called each game tick before collision detection.
        """
        self.position += self.displacement

    def check_collisions(self):
        """! Check for game-ending collisions.
        
        Detects two types of collisions:
        1. Wall collisions: Snake head touches screen boundaries
        2. Self collisions: Snake head touches any part of its body
        
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

        # Check dynamic wall collisions
        for wall in self.walls:
            if np.array_equal(self.position, wall):
                self.game_over = True
                print("Game over: Dynamic wall collision")
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

    def create_wall(self):
        """Create a random wall and remove the oldest when needed"""

        available_positions = []

        for x in range(0, self.SCREEN_WIDTH, self. SNAKE_BLOCK):
            for y in range(0,self.SCREEN_HEIGHT, self.SNAKE_BLOCK):
                position = [x,y]

                # avoid snake, food, and eixisting walls
                if (not np.array_equal(position, self.position)
                    and not any(np.array_equal(position, segment)
                                for segment in self.snake_list)
                    and not np.array_equal(position, self.food_position)
                    and position not in self.walls):

                    available_positions.append(position)
        
        # Remove oldest wall if maximum reached
        if len(self.walls) >= self.max_walls:
            self.walls.pop(0)

        # Add a new wall at a random available position
        if available_positions:
            new_wall = random.choice(available_positions)
            self.walls.append(new_wall)


    def create_food(self):
        """! Generate new food at random position.
        """
        available_positions = []

        for x in range(0, self.SCREEN_WIDTH, self.SNAKE_BLOCK):
            for y in range(0, self.SCREEN_HEIGHT, self.SNAKE_BLOCK):
                position = [x, y]

                # Avoid snake head, body, and dynamic walls
                if (not np.array_equal(position, self.position)
                    and not any(np.array_equal(position, segment)
                                for segment in self.snake_list)
                    and position not in self.walls):

                    available_positions.append(position)

        if available_positions:
            self.food_position = np.array(
                random.choice(available_positions)
            )
        else:
            # No available space remaining
            self.food_position = None
            self.game_over = True

    def eat_food(self):
        """! Check for food consumption and handle growth.
        
        Performs these operations:
        1. Checks if snake head overlaps with food position
        2. Increases snake length if food is eaten
        3. Generates new food position after consumption
        4. Updates is_eating flag for reward calculation
        """
        self.is_eating = np.array_equal(self.position, self.food_position)
        if self.is_eating:
            self.snake_length += 1
            self.create_food()
            self.create_wall()

    def process(self):
        """! Process one complete game tick.
        
        Executes main game loop operations in order:
        1. Updates snake head position
        2. Checks for collisions
        3. Updates snake body segments
        4. Handles food consumption
        
        This method should be called once per game tick/frame.
        """
        self.update_snake_position()  # Move snake head
        self.check_collisions()       # Check for game-ending conditions
        self.update_snake_list()      # Update body positions
        self.eat_food()               # Handle food interactions
