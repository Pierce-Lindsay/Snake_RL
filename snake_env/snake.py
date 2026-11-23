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
        self.game_won = False
        self.is_eating = False  #!< Tracks if snake is currently eating food
        
        # Snake properties
        self.snake_list = []    #!< List of coordinates for each snake segment
        self.snake_length = 1   #!< Current length of the snake
        self.walls = []
        self.max_walls = 3
        self.wall_length = 3
        
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
            for cell in wall:
                if np.array_equal(self.position, cell):
                    self.game_over = True
                    print("Game over: Dynamic wall collision")
                    break

            if self.game_over:
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
        possible_walls = []

        # Get all occupied positions
        occupied = (
            self.snake_list
            + [self.position.tolist()]
            + [self.food_position.tolist()]
            + [cell for wall in self.walls for cell in wall]
        )

        for x in range(0, self.SCREEN_WIDTH, self.SNAKE_BLOCK):
            for y in range(0, self.SCREEN_HEIGHT, self.SNAKE_BLOCK):

                for orientation in ["horizontal", "vertical"]:

                    if orientation == "horizontal":
                        wall = [
                            [x + i * self.SNAKE_BLOCK, y]
                            for i in range(self.wall_length)
                        ]
                    else:
                        wall = [
                            [x, y + i * self.SNAKE_BLOCK]
                            for i in range(self.wall_length)
                        ]

                    # Keep walls inside screen boundaries
                    if any(
                        wx >= self.SCREEN_WIDTH or wy >= self.SCREEN_HEIGHT
                        for wx, wy in wall
                    ):
                        continue

                    # Prevent overlap with snake, food, or other walls
                    if any(cell in occupied for cell in wall):
                        continue

                    possible_walls.append(wall)

        if not possible_walls:
            return

        # Remove oldest wall when maximum is reached
        if len(self.walls) >= self.max_walls:
            self.walls.pop(0)

        # Add a randomly selected wall
        self.walls.append(random.choice(possible_walls))


    def create_food(self):
        """Generate food at a random unoccupied grid position."""
        available_positions = []

        # Collect all occupied wall cells
        wall_cells = [
            cell
            for wall in self.walls
            for cell in wall
        ]

        for x in range(0, self.SCREEN_WIDTH, self.SNAKE_BLOCK):
            for y in range(0, self.SCREEN_HEIGHT, self.SNAKE_BLOCK):
                position = [x, y]

                # Avoid snake head, body, and all wall cells
                if (not np.array_equal(position, self.position)
                    and not any(np.array_equal(position, segment)
                                for segment in self.snake_list)
                    and position not in wall_cells):

                    available_positions.append(position)

        if available_positions:
            self.food_position = np.array(
                random.choice(available_positions)
            )
        else:
            self.food_position = None
            self.game_won = True
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
