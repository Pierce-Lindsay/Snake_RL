#class for handling a fruit with bouncing logic off walls and the snake when it doesn't collide with the head
import pygame
import numpy as np
import collision

class Bouncing_Fruit:
    def __init__(self, pos: np.ndarray, velocity: np.ndarray):
        self.velocity = velocity
        self.pos = pos
    def update(self, snake):
        new_pos = self.pos + self.velocity
        snake_collision_index = collision.check_snake_collision(new_pos, snake.snake_list)
        wall_collisions =  collision.check_wall_collision(new_pos)
        if wall_collisions[0]:
            self.velocity[0] *= -1
        if wall_collisions[1]:
            self.velocity[1] *= -1              
        if(snake_collision_index and snake_collision_index != 0):
            self.velocity *= -1
            
        self.pos += self.velocity
    def draw(self):
        pass