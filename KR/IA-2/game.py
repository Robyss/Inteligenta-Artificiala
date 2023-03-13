import pygame as pg
import sys
import os
import time
from tictactoe import TicTacToe

vec2 = pg.math.Vector2


class Game:
    FPS = 240
    clock = pg.time.Clock()
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    ORANGE = (229, 180, 124)

    def __init__(self, window, n, k1, k2, gamemode=0, bot_player=0, depth=3, alpha_beta=True, joc=None):
        pg.init()
        self.WIN = window
        pg.display.set_caption("Roby's Tic Tac Toe on Steroids!")
        self.N = n
        self.K1 = k1
        self.K2 = k2
        self.WIDTH = window.get_width()
        self.HEIGHT = window.get_height()
        self.PIECE_SIZE = (self.HEIGHT - 100) // self.N


        self.gamemode = gamemode
        self.bot_player = bot_player
        self.depth = depth
        self.alpha_beta = alpha_beta

        if joc is None:     # Pentru load game
            self.joc = TicTacToe(n, k1, k2, gamemode)
        else:
            self.joc = joc
        self.background = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "background.png")),
                                                   (self.WIDTH, self.HEIGHT))
        self.X_image = self.load_texture(os.path.join("assets", "X_frumos_mid.png"))
        self.O_image = self.load_texture(os.path.join("assets", "0_frumos_mid.png"))
        self.X_image_punct = self.load_texture(os.path.join("assets", "X_corect_mid.png"))
        self.O_image_punct = self.load_texture(os.path.join("assets", "0_corect_mid.png"))
        self.hint = self.load_texture(os.path.join("assets", "hint.png"))

        self.font = pg.font.Font("assets/Game_of_Steroids.ttf", 30)
        self.small_font = pg.font.Font("assets/Game_of_Steroids.ttf", 20)

        self.flag = True

    def dict(self):
        return {"N": self.N, "K1": self.K1, "K2": self.K2,
                "gamemode": self.gamemode,
                "bot_player": self.bot_player,
                "depth": self.depth,
                "alpha_beta": self.alpha_beta,
                "joc": self.joc.dict()
                }

    def load_texture(self, path):
        return pg.transform.smoothscale(pg.image.load(path), (self.PIECE_SIZE, self.PIECE_SIZE))

    """ ########## DRAWINGS ##########"""

    def draw_grid(self, win, piece_size, n, border_size):
        for i in range(n + 1):
            pg.draw.line(win, self.ORANGE, (0, i * piece_size), (piece_size * n, i * piece_size), border_size)
            pg.draw.line(win, self.ORANGE, (i * piece_size, 0), (i * piece_size, piece_size * n), border_size)

    def draw_objects(self):

        for i, row in enumerate(self.joc.grid):
            for j, obj in enumerate(row):
                if obj == 1:
                    if self.joc.move_is_point(i, j):
                        self.WIN.blit(self.X_image_punct, (j * self.PIECE_SIZE, i * self.PIECE_SIZE))
                    else:
                        self.WIN.blit(self.X_image, (j * self.PIECE_SIZE, i * self.PIECE_SIZE))
                elif obj == 2:
                    if self.joc.move_is_point(i, j):
                        self.WIN.blit(self.O_image_punct, (j * self.PIECE_SIZE, i * self.PIECE_SIZE))
                    else:
                        self.WIN.blit(self.O_image, (j * self.PIECE_SIZE, i * self.PIECE_SIZE))

        self.hint.set_alpha(50)
        N = self.N
        moves = self.joc.moves
        if len(moves) < N * N - 1:
            for i in range(len(moves)):
                self.WIN.blit(self.hint, (moves[i][1] * self.PIECE_SIZE, moves[i][0] * self.PIECE_SIZE))

    def draw_text(self):
        WIDTH, HEIGHT = self.WIN.get_size()
        player_x_points = self.small_font.render(f"Player X: {self.joc.points_x}", True, 'purple')
        player_0_points = self.small_font.render(f"Player 0: {self.joc.points_0}", True, 'purple')

        right_text = self.small_font.render("", True, 'purple')
        if not self.joc.game_ended:
            right_text = self.small_font.render(f"Time: {round(time.time() - self.joc.time_since_last_move,1)}", True, self.ORANGE)
            if self.gamemode == 1:
                if self.bot_player == self.joc.player:
                    right_text = self.small_font.render("Bot is thinking...", True, self.ORANGE)
            elif self.gamemode == 2:
                right_text = self.small_font.render("Bot is thinking...", True, self.ORANGE)

        if not self.joc.game_ended:
            bottom_text = self.font.render(f"Player {self.joc.get_player()} turn", True, 'purple')
        else:
            if self.joc.points_x == self.joc.points_0:
                bottom_text = self.font.render("It's a Draw!", True, 'purple')
            else:
                bottom_text = self.font.render(f"Player {self.joc.winner} won!", True, 'purple')

        self.WIN.blit(bottom_text, (WIDTH // 2 - bottom_text.get_width() // 2, HEIGHT - bottom_text.get_height() - 10))
        self.WIN.blit(player_x_points, (10, HEIGHT - player_x_points.get_height() - 10))
        self.WIN.blit(player_0_points, (WIDTH - player_0_points.get_width() - 10, HEIGHT - player_0_points.get_height() - 10))
        self.WIN.blit(right_text, (WIDTH - right_text.get_width() - 10, 10))

    def draw_window(self):
        self.WIN.fill((255, 255, 255))
        self.WIN.blit(self.background, (0, 0))
        self.draw_grid(self.WIN, self.PIECE_SIZE, self.N, 3)
        self.draw_objects()
        self.draw_text()
        pg.display.update()

    "##################################################"

    def clicked_move(self, MOUSE_POS):
        current_cell = vec2(MOUSE_POS) // self.PIECE_SIZE
        col, row = map(int, current_cell)
        self.joc.move_player(row, col)

    def click_debug(self, MOUSE_POS):
        current_cell = vec2(MOUSE_POS) // self.PIECE_SIZE
        col, row = map(int, current_cell)
        self.joc.grid[row][col] = self.joc.player
        print(f"Debug: {row}, {col}")
        print(self.joc)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(0)
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    self.clicked_move(pg.mouse.get_pos())
                if event.button == 3:
                    self.click_debug(pg.mouse.get_pos())

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_c:
                    self.flag = not self.flag
                elif event.key == pg.K_SPACE:
                    self.pause_screen()
                elif event.key == pg.K_r:
                    self.joc = TicTacToe(self.N, self.K1, self.K2, self.gamemode)
                elif event.key == pg.K_s:
                    self.save_game()
                elif event.key == pg.K_u:
                    if self.gamemode == 0:
                        self.joc.undo_player()
                    else:
                        self.joc.undo_ai_player()
                elif event.key == pg.K_n:
                    if self.gamemode == 1:
                        self.joc.ai_change_move()

    def pause_screen(self):
        time_spent_paused = time.time()
        gamemode_text = None
        if self.gamemode == 0:
            gamemode_text = self.font.render("Player vs Player", True, 'purple')
        elif self.gamemode == 1:
            gamemode_text = self.font.render("Player vs AI", True, 'purple')
        elif self.gamemode == 2:
            gamemode_text = self.font.render("AI vs AI", True, 'purple')

        moves_x_text = self.small_font.render(f"X Moves: {self.joc.player_x_moves}", True, 'purple')
        moves_0_text = self.small_font.render(f"0 Moves: {self.joc.player_0_moves}", True, 'purple')


        pause_text = self.font.render("Game paused", True, 'purple')
        time_spent = self.small_font.render(f"Time spent: {round(time.time() - self.joc.start_time, 2)} seconds", True, 'purple')

        if self.joc.game_ended:
            time_spent = self.small_font.render(f"Time spent: {self.joc.end_time} seconds", True, 'purple')
            pause_text = self.font.render("Game ended", True, 'purple')
        nodes_generated = self.small_font.render("", True, 'purple')
        if self.gamemode != 0:
            nodes_generated = self.small_font.render(f"Nodes generated: {sum(self.joc.nodes_generated)}", True, 'purple')

        pause = True
        while pause:
            self.clock.tick(self.FPS)
            self.WIN.fill((255, 255, 255))
            self.WIN.blit(gamemode_text, (self.WIDTH // 2 - gamemode_text.get_width() // 2, 20))

            self.WIN.blit(pause_text, (self.WIDTH // 2 - pause_text.get_width() // 2,
                                       self.HEIGHT // 2 - pause_text.get_height() // 2))
            self.WIN.blit(moves_x_text, (self.WIDTH // 2 - moves_x_text.get_width() // 2 - 100,
                                          self.HEIGHT // 2 - moves_x_text.get_height() // 2 + 50))
            self.WIN.blit(moves_0_text, (self.WIDTH // 2 - moves_0_text.get_width() // 2 + 100,
                                          self.HEIGHT // 2 - moves_0_text.get_height() // 2 + 50))
            self.WIN.blit(time_spent, (self.WIDTH // 2 - time_spent.get_width() // 2,
                                       self.HEIGHT // 2 - time_spent.get_height() // 2 + 100))
            self.WIN.blit(nodes_generated, (self.WIDTH // 2 - nodes_generated.get_width() // 2,
                                            self.HEIGHT // 2 - nodes_generated.get_height() // 2 + 150))

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit(0)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        pause = False
            pg.display.update()

        self.joc.start_time += time.time() - time_spent_paused
        self.joc.time_since_last_move += time.time() - time_spent_paused

    def save_game(self):
        print("Saving game...")
        data = self.dict()
        import json
        folder = "saves"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = "save" + str(len(os.listdir(folder))) + ".txt"
        with open(os.path.join(folder, filename), "w") as f:
            json.dump(data, f)
        print("Game saved!")

    @staticmethod
    def load_game(WIN, data):
        return Game(WIN, data["N"], data["K1"], data["K2"],
                    data["gamemode"], data["bot_player"],
                    data["depth"], data["alpha_beta"],
                    TicTacToe.load_game(data["joc"]))


    def run(self):
        while True:
            self.clock.tick(self.FPS)
            self.check_events()  # Events

            self.draw_window()  # afisam fereastra

            if self.gamemode == 1:
                if not self.joc.game_ended:
                    if self.joc.player == 2 and self.bot_player == 2:
                        self.joc.ai_move(False, self.depth, self.alpha_beta)
                    elif self.joc.player == 1 and self.bot_player == 1:
                        self.joc.ai_move(True, self.depth, self.alpha_beta)
            if self.gamemode == 2:
                if not self.joc.game_ended:
                    if self.joc.player == 2:
                        self.joc.ai_move(False, self.depth)

                    elif self.joc.player == 1:

                        self.joc.ai_move(True, self.depth)



if __name__ == "__main__":
    window = pg.display.set_mode((800, 600))
    N, K1, K2 = 6, 3, 3
    game = Game(window, N, K1, K2, gamemode=1, bot_player=2, depth=3, alpha_beta=True)
    game.run()
