import unittest

import numpy as np

from snake_env.snake import Snake


class NumberedFruitTests(unittest.TestCase):
    def test_initial_fruits_are_distinct_and_avoid_snake_head(self):
        game = Snake(numbered_fruits=True, fruit_count=5)

        positions = [tuple(position) for position in game.fruit_positions]
        self.assertEqual(len(positions), 5)
        self.assertEqual(len(set(positions)), 5)
        self.assertNotIn(tuple(game.position), positions)
        np.testing.assert_array_equal(game.food_position, game.fruit_positions[0])

    def test_eating_one_shifts_remaining_fruits_and_adds_a_new_five(self):
        game = Snake(numbered_fruits=True)
        original_positions = [position.copy() for position in game.fruit_positions]
        game.position = original_positions[0].copy()
        game.snake_list = [game.position.tolist()]

        game.eat_food()

        self.assertTrue(game.is_eating)
        self.assertEqual(game.snake_length, 2)
        for shifted, previous in zip(game.fruit_positions[:4], original_positions[1:]):
            np.testing.assert_array_equal(shifted, previous)
        self.assertNotIn(
            tuple(game.fruit_positions[4]),
            {tuple(position) for position in game.snake_list + game.fruit_positions[:4]},
        )
        np.testing.assert_array_equal(game.food_position, game.fruit_positions[0])

    def test_selected_fruit_count_is_used_and_preserved_after_eating(self):
        for count in (3, 4, 5):
            with self.subTest(count=count):
                game = Snake(numbered_fruits=True, fruit_count=count)
                original_positions = [
                    position.copy() for position in game.fruit_positions
                ]

                self.assertEqual(len(game.fruit_positions), count)
                game.position = original_positions[0].copy()
                game.snake_list = [game.position.tolist()]
                game.eat_food()

                self.assertEqual(len(game.fruit_positions), count)
                for shifted, previous in zip(
                    game.fruit_positions[:-1], original_positions[1:]
                ):
                    np.testing.assert_array_equal(shifted, previous)

    def test_fruit_count_rejects_unsupported_values(self):
        for count in (2, 6):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    Snake(numbered_fruits=True, fruit_count=count)

    def test_selecting_count_enables_numbered_fruits(self):
        game = Snake()

        game.set_numbered_fruits(True, 4)

        self.assertTrue(game.numbered_fruits)
        self.assertEqual(game.fruit_count, 4)
        self.assertEqual(len(game.fruit_positions), 4)

    def test_count_cannot_be_changed_during_active_game(self):
        game = Snake(numbered_fruits=True, fruit_count=3)
        game.process()

        game.set_numbered_fruits(True, 5)

        self.assertEqual(game.fruit_count, 3)
        self.assertEqual(len(game.fruit_positions), 3)

    def test_touching_a_fruit_above_one_ends_the_game(self):
        game = Snake(numbered_fruits=True)
        game.position = np.array([100.0, 100.0])
        game.displacement = np.array([10.0, 0.0])
        game.snake_list = [[90.0, 100.0]]
        game.fruit_positions = [
            np.array([30.0, 30.0]),
            np.array([110.0, 100.0]),
            np.array([130.0, 100.0]),
            np.array([150.0, 100.0]),
            np.array([170.0, 100.0]),
        ]
        game.food_position = game.fruit_positions[0]

        game.process()

        self.assertTrue(game.game_over)
        self.assertFalse(game.is_eating)
        self.assertEqual(game.snake_length, 1)

    def test_numbered_fruits_are_disabled_by_default(self):
        game = Snake()

        self.assertFalse(game.numbered_fruits)
        self.assertEqual(len(game.fruit_positions), 1)
        np.testing.assert_array_equal(game.food_position, game.fruit_positions[0])

    def test_numbered_fruits_can_be_toggled_off(self):
        game = Snake(numbered_fruits=True)
        enabled = game.toggle_numbered_fruits()

        self.assertFalse(enabled)
        self.assertFalse(game.numbered_fruits)
        self.assertEqual(len(game.fruit_positions), 1)

    def test_numbered_fruits_cannot_be_toggled_during_active_game(self):
        game = Snake(numbered_fruits=True)
        game.process()

        enabled = game.toggle_numbered_fruits()

        self.assertTrue(enabled)
        self.assertTrue(game.numbered_fruits)


if __name__ == "__main__":
    unittest.main()
