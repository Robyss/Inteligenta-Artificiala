from game import Game
import pygame as pg
import os
import sys
from button import Button

WIDTH = 800
HEIGHT = 600


class GameState:
    state = "intro"
    FPS = 144
    clock = pg.time.Clock()
    N, K1, K2 = None, None, None
    n, k1, k2 = 5, 2, 2
    
    pos = 50
    string = "Choose the N number of rows/columns"
    string_1 = "Between 4 and 10 (F10 for 10)"

    load_game_string = ""

    flag_algo_minmax, flag_algo_alphabeta, flag_incepator, flag_mediu, flag_avansat = False, False, False, False, False
    flag_X = True
    load_error = ""
    gamemode = 0

    ORANGE = (255, 180, 124)

    def __init__(self) -> None:
        pg.init()
        self.WIN = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Roby's Tic Tac Toe on Steroids!")

        self.input_box = pg.Surface((400, 50))
        self.input_box.fill((255, 255, 255))
        self.input_box_rect = self.input_box.get_rect(center=(WIDTH // 2, HEIGHT // 2))

        self.background = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "background.png")), (WIDTH, HEIGHT))

        self.button_image = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "button.png")), (400, 100))
        self.button_image_mic = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "button.png")), (200, 100))

        self.button_keep = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "buton_aqua.png")), (400, 100))
        self.button_keep_mic = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "buton_aqua.png")), (200, 100))
        self.button_clicked = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "buton_mov.png")), (400, 100))
        self.button_clicked_mic = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "buton_mov.png")), (200, 100))

        self.buton_image_rosu_mic = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "buton_rosu.png")), (200, 100))
        self.buton_image_verde_mic = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "buton_verde.png")), (200, 100))
        self.buton_image_portocaliu_mic = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "buton_portocaliu.png")), (200, 100))

        self.font = pg.font.Font("assets/Game_of_Steroids.ttf", 30)
        self.title_font = pg.font.Font("assets/Game_of_Steroids.ttf", 46)

        self.text_color = 'white'
        self.text_hover_color = self.ORANGE

        """ Butoanele statice"""
        self.buton_min_max = Button(self.button_image, (200, 150), "Min-max", self.font, self.text_color, self.text_hover_color, self.button_clicked)
        self.buton_alpha_beta = Button(self.button_image, (600, 150), "Alpha-Beta", self.font, self.text_color, self.text_hover_color, self.button_clicked)
        self.buton_incepator = Button(self.buton_image_verde_mic, (200, 300), "Incepator", self.font, self.text_color,  self.text_hover_color, self.button_clicked_mic)
        self.buton_mediu = Button(self.buton_image_portocaliu_mic, (400, 300), "Mediu", self.font, self.text_color, self.text_hover_color, self.button_clicked_mic)
        self.buton_avansat = Button(self.buton_image_rosu_mic, (600, 300), "Avansat", self.font, self.text_color, self.text_hover_color, self.button_clicked_mic)
        """ """

        self.image_x = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "x_frumos.png")), (100, 100))
        self.image_0 = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "0_frumos.png")), (100, 100))
        self.image_x_verde = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "x_corect.png")), (100, 100))
        self.image_0_verde = pg.transform.smoothscale(pg.image.load(os.path.join("assets", "0_corect.png")), (100, 100))

    def state_manager(self):
        if self.state == "intro":
            self.intro()
        if self.state == "load_game":
            self.load_game()
        elif self.state == "choose_rules":
            self.choose_rules()
        elif self.state == "choose_player":
            self.choose_player()
        elif self.state == "choose_fighter":
            self.choose_fighter()
        elif self.state == "choose_algo":
            self.choose_algo()
        elif self.state == "game":
            self.game()

    def handle_load_game(self):
        files = os.listdir("saves")

        if len(files) == 0:
            self.load_error = "No local saves!"
            return
        elif self.load_game_string not in files:
            self.load_error = "File not found!"
        else:
            print("Loading game...")
            self.load_error = ""
            self.initialize_load_game()

    def initialize_load_game(self):
        import json
        with open(os.path.join("saves", self.load_game_string), "r") as f:
            data = json.load(f)

        joc = Game.load_game(self.WIN, data)
        joc.run()


    def intro(self):
        self.WIN.fill((0, 0, 0))
        self.WIN.blit(self.background, (0, 0))

        title = self.title_font.render("Tic Tac Toe on Steroids!", True, self.ORANGE)
        self.WIN.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))

        MOUSE_POS = pg.mouse.get_pos()
        buton_new_game = Button(self.button_image, (200, 300), "New Game", self.font, self.text_color, self.text_hover_color,
                            self.button_keep)
        buton_load_game = Button(self.button_image, (600, 300), "Load Game", self.font, self.text_color, self.text_hover_color,
                            self.button_keep)
        buton_quit = Button(self.button_image, (400, 400), "Quit", self.font, self.text_color, self.text_hover_color,
                            self.button_keep)

        for buton in [buton_new_game, buton_quit, buton_load_game]:
            if pg.mouse.get_pressed()[0]:
                buton.changeClickedImage(MOUSE_POS)
            buton.changeColor(MOUSE_POS)
            buton.update(self.WIN)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(0)
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    if buton_new_game.checkForInput(MOUSE_POS):
                        self.state = "choose_rules"
                    if buton_load_game.checkForInput(MOUSE_POS):
                        self.load_game_string = ""
                        self.state = "load_game"
                    elif buton_quit.checkForInput(MOUSE_POS):
                        print("Pussy")
                        pg.quit()
                        sys.exit(0)

    def load_game(self):
        self.WIN.fill((0, 0, 0))
        self.WIN.blit(self.background, (0, 0))
        normal_font = pg.font.SysFont("Lucida Console", 30)  # Merge bine pe Windows
        load_file_name = normal_font.render(f"{self.load_game_string}", True, 'purple')
        text = self.font.render("Write down the file name", True, self.ORANGE)
        error_text = self.font.render(f"{self.load_error}", True, 'red')
        buton_inapoi = Button(self.button_image, (400, 500), "Back", self.font, self.text_color, self.text_hover_color,
                              self.button_keep)

        if self.load_error:
            self.WIN.blit(error_text, (WIDTH // 2 - error_text.get_width() // 2, 400))
        self.WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, 50))
        self.WIN.blit(self.input_box, (WIDTH // 2 - self.input_box.get_width() // 2, 200))

        MOUSE_POS = pg.mouse.get_pos()

        if pg.mouse.get_pressed()[0]:
            buton_inapoi.changeClickedImage(MOUSE_POS)
        buton_inapoi.changeColor(MOUSE_POS)
        buton_inapoi.update(self.WIN)


        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(0)
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    if buton_inapoi.checkForInput(MOUSE_POS):
                        self.load_game_string = ""
                        self.load_error = ""
                        self.state = "intro"
            elif event.type == pg.KEYDOWN:
                if event.unicode == "\r":   # Enter
                    self.handle_load_game()
                    self.load_game_string = ""
                elif event.unicode == "\x08":   # Backspace
                    self.load_game_string = self.load_game_string[:-1]
                else:
                    self.load_game_string += event.unicode

        center_text = load_file_name.get_rect()
        center_text.center = self.input_box.get_rect().center

        # Update the context of the input box
        self.input_box.fill((255, 255, 255))
        self.input_box.blit(load_file_name, center_text)


    def choose_rules(self):
        self.WIN.fill((0, 0, 0))
        self.WIN.blit(self.background, (0, 0))

        text = self.font.render(self.string, True, self.ORANGE)
        text_interval = self.font.render(self.string_1, True, 'white')
        help = self.font.render("Press a number from your keyboard", True, 'black')
        self.WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, self.pos))

        self.WIN.blit(text_interval, (WIDTH // 2 - text_interval.get_width() // 2, self.pos + 50))
        self.WIN.blit(help, (WIDTH // 2 - help.get_width() // 2, 550))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(0)
            elif event.type == pg.KEYDOWN:
                if event.unicode.isnumeric() or event.key == pg.K_F10:
                    if self.N is None:
                        if event.key == pg.K_F10 or 4 <= int(event.unicode) < 10:   # daca inversezi conditia, eroare :D
                            self.N = 10 if event.key == pg.K_F10 else int(event.unicode)
                            self.string = "You chose N=" + str(self.N) + "   Now choose K1"
                            self.string_1 = "Between 0 and " + str(self.N - 1)
                            self.pos += 100
                            continue
                    if self.N is not None and self.K1 is None:
                        if 0 <= int(event.unicode) < self.N:
                            self.K1 = int(event.unicode)
                            self.string = "You chose K1=" + str(self.K1) + "   Now choose K2"
                            self.string_1 = "Between 0 and " + str(self.N - 1)
                            self.pos += 100
                            continue

                    if self.N is not None and self.K1 is not None and self.K2 is None:
                        if self.K1 + int(event.unicode) == 0:
                            self.string = "K1 + K2 must be greater than 0"
                            self.string_1 = "Choose K2 > 0"
                            continue

                        if 0 <= int(event.unicode) < self.N:
                            self.K2 = int(event.unicode)
                            self.string = "Choose the N number of rows/columns"
                            self.string_1 = "Between 4 and 10 (F10 for 10)"
                            self.pos = 50
                            self.n, self.k1, self.k2 = self.N, self.K1, self.K2
                            self.N, self.K1, self.K2 = None, None, None
                            self.state = "choose_player"

    def choose_player(self):

        buton_juc_juc = Button(self.button_image, (400, 200), "Jucator vs Jucator", self.font, self.text_color, self.text_hover_color, self.button_keep)
        buton_juc_ai = Button(self.button_image, (400, 300), "Jucator vs AI", self.font, self.text_color, self.text_hover_color, self.button_keep)
        buton_ai_ai = Button(self.button_image, (400, 400), "AI vs AI", self.font, self.text_color, self.text_hover_color, self.button_keep)
        buton_inapoi = Button(self.button_image, (400, 500), "Back", self.font, self.text_color, self.text_hover_color, self.button_keep)

        MOUSE_POS = pg.mouse.get_pos()

        self.WIN.fill((0, 0, 0))
        self.WIN.blit(self.background, (0, 0))

        for buton in [buton_juc_juc, buton_juc_ai, buton_ai_ai, buton_inapoi]:
            if pg.mouse.get_pressed()[0]:
                buton.changeClickedImage(MOUSE_POS)
            buton.changeColor(MOUSE_POS)
            buton.update(self.WIN)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(0)
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    if buton_juc_ai.checkForInput(MOUSE_POS):
                        self.gamemode = 1
                        self.state = "choose_fighter"
                    elif buton_juc_juc.checkForInput(MOUSE_POS):
                        self.gamemode = 0
                        self.state = "game"
                    elif buton_ai_ai.checkForInput(MOUSE_POS):
                        self.gamemode = 2
                        self.state = "game"
                    elif buton_inapoi.checkForInput(MOUSE_POS):
                        self.state = "choose_rules"

    def choose_fighter(self):
        self.WIN.fill((0, 0, 0))
        self.WIN.blit(self.background, (0, 0))
        text = self.font.render("Choose your fighter", True, (255, 255, 255))
        self.WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, 30))

        MOUSE_POS = pg.mouse.get_pos()

        buton_x = Button(self.image_x, (250, 250), "", self.font, self.text_color, self.text_hover_color, self.image_x_verde)
        buton_0 = Button(self.image_0, (550, 250), "", self.font, self.text_color, self.text_hover_color, self.image_0_verde)
        buton_back = Button(self.button_image_mic, (400, 500), "Back", self.font, self.text_color, self.text_hover_color, self.button_keep_mic)

        for buton in [buton_x, buton_0, buton_back]:
            if pg.mouse.get_pressed()[0]:
                buton.changeClickedImage(MOUSE_POS)
            buton.changeColor(MOUSE_POS)
            buton.update(self.WIN)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(0)
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    if buton_x.checkForInput(MOUSE_POS):
                        self.flag_X = True
                        self.state = "choose_algo"
                    elif buton_0.checkForInput(MOUSE_POS):
                        self.flag_X = False
                        self.state = "choose_algo"
                    elif buton_back.checkForInput(MOUSE_POS):
                        self.state = "choose_player"
                        self.flag_X = False


    def choose_algo(self):
        self.WIN.fill((0, 0, 0))
        self.WIN.blit(self.background, (0, 0))
        text = self.font.render("Choose opponent algorithm", True, (255, 255, 255))
        self.WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, 30))

        MOUSE_POS = pg.mouse.get_pos()

        buton_min_max = self.buton_min_max
        buton_alpha_beta = self.buton_alpha_beta

        buton_incepator = self.buton_incepator
        buton_mediu = self.buton_mediu
        buton_avansat = self.buton_avansat

        buton_confirm = Button(self.button_image_mic, (300, 500), "Confirm", self.font, self.text_color, self.text_hover_color, self.button_clicked_mic)
        buton_back = Button(self.button_image_mic, (500, 500), "Back", self.font, self.text_color, self.text_hover_color, self.button_clicked_mic)


        if pg.mouse.get_pressed()[0]:
            if buton_confirm.checkForInput(MOUSE_POS):
                buton_confirm.changeClickedImage(MOUSE_POS)
            elif buton_back.checkForInput(MOUSE_POS):
                buton_back.changeClickedImage(MOUSE_POS)

        for buton in [buton_min_max, buton_alpha_beta, buton_incepator, buton_mediu, buton_avansat, buton_confirm, buton_back]:
            buton.changeColor(MOUSE_POS)
            buton.update(self.WIN)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(0)
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:
                    if buton_confirm.checkForInput(MOUSE_POS):
                        if (self.flag_algo_alphabeta or self.flag_algo_minmax) \
                                and (self.flag_incepator or self.flag_mediu or self.flag_avansat):
                            self.state = "game"
                    elif buton_back.checkForInput(MOUSE_POS):
                        self.state = "choose_fighter"

                    elif buton_alpha_beta.checkForInput(MOUSE_POS):
                        if not self.flag_algo_alphabeta:
                            buton_alpha_beta.switchImage()
                            self.flag_algo_alphabeta = True
                        if self.flag_algo_minmax:
                            self.flag_algo_minmax = False
                            buton_min_max.switchImage()
                    elif buton_min_max.checkForInput(MOUSE_POS):
                        if not self.flag_algo_minmax:
                            self.flag_algo_minmax = True
                            buton_min_max.switchImage()
                        if self.flag_algo_alphabeta:
                            self.flag_algo_alphabeta = False
                            buton_alpha_beta.switchImage()

                    elif buton_incepator.checkForInput(MOUSE_POS):
                        if not self.flag_incepator:
                            self.flag_incepator = True
                            buton_incepator.switchImage()
                        if self.flag_mediu:
                            self.flag_mediu = False
                            buton_mediu.switchImage()
                        if self.flag_avansat:
                            self.flag_avansat = False
                            buton_avansat.switchImage()

                    elif buton_mediu.checkForInput(MOUSE_POS):
                        if not self.flag_mediu:
                            self.flag_mediu = True
                            buton_mediu.switchImage()
                        if self.flag_incepator:
                            self.flag_incepator = False
                            buton_incepator.switchImage()
                        if self.flag_avansat:
                            self.flag_avansat = False
                            buton_avansat.switchImage()

                    elif buton_avansat.checkForInput(MOUSE_POS):
                        if not self.flag_avansat:
                            self.flag_avansat = True
                            buton_avansat.switchImage()
                        if self.flag_incepator:
                            self.flag_incepator = False
                            buton_incepator.switchImage()
                        if self.flag_mediu:
                            self.flag_mediu = False
                            buton_mediu.switchImage()

    def game(self):
        bot_player = 2 if self.flag_X else 1
        depth = 3   # default depth for ai vs ai
        algo = False if self.flag_algo_minmax else True
        if self.flag_incepator:
            depth = 1
        elif self.flag_mediu:
            depth = 2
        elif self.flag_avansat:
            depth = 4
        print("depth: ", depth)

        joc = Game(self.WIN, self.n, self.k1, self.k2, self.gamemode, bot_player, depth, algo)
        joc.run()

    def game_load(self):
        joc = self.initialize_load_game()
        joc.run()

    def run(self):

        while True:
            self.clock.tick(self.FPS)
            self.state_manager()
            pg.display.update()


if __name__ == "__main__":
    game = GameState()
    game.run()
