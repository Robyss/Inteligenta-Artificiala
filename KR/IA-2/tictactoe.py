from typing import List, Tuple
import time
import statistics
import copy
import numpy as np



class TicTacToe:
    """ Constante """
    X = 1
    O = 2

    def __init__(self, n, k1, k2, gamemode=0):
        self.N = n
        self.K1 = k1
        self.K2 = k2
        self.grid = [[0 for _ in range(n)] for _ in range(n)]
        self.player = 1
        self.points_x = 0
        self.points_0 = 0
        self.player_x_moves = 0
        self.player_0_moves = 0
        self.moves = self.generate_moves(self.player)  # Stocheaza mutarile posibile ale unui jucator
        self.last_moves = []  # Stocheaza istoricul mutarilor
        self.ai_moves = []
        self.last_ai_move = None
        self.points_position = []
        self.game_ended = False
        self.winner = None
        self.gamemode = gamemode    # Folosit pentru anumite afisari in consola
        self.stare = None

        # Statistics
        self.start_time = time.time()
        self.time_since_last_move = time.time()
        self.player_move_times = []
        self.ai_move_times = []
        self.end_time = 0

        self.nodes_generated = []
        self.min_nodes_generated = float('inf')
        self.max_nodes_generated = 0
        self.mean_nodes_generated = 0
        self.median_nodes_generated = 0
        print(self)

    def dict(self):  # Ca sa implementez __dict__ trebuie sa scriu un bloc la fel de mare si pt __deepcopy__
        return {
            'N': self.N,
            'K1': self.K1,
            'K2': self.K2,
            'grid': self.grid,
            'player': self.player,
            'points_x': self.points_x,
            'points_0': self.points_0,
            'player_x_moves': self.player_x_moves,
            'player_0_moves': self.player_0_moves,
            'moves': self.moves,
            'last_moves': self.last_moves,
            # 'ai_moves': self.ai_moves,
            'last_ai_move': self.last_ai_move,
            'points_position': self.points_position,
            'game_ended': self.game_ended,
            'winner': self.winner,
            'gamemode': self.gamemode,
            'start_time': self.start_time,
            'time_since_last_move': self.time_since_last_move,
            'player_move_times': self.player_move_times,
            'ai_move_times': self.ai_move_times,
            'end_time': self.end_time,
            'nodes_generated': self.nodes_generated,
            'min_nodes_generated': self.min_nodes_generated,
            'max_nodes_generated': self.max_nodes_generated,
            'mean_nodes_generated': self.mean_nodes_generated,
            'median_nodes_generated': self.median_nodes_generated
        }

    @staticmethod
    def load_game(data):
        joc = TicTacToe(data['N'], data['K1'], data['K2'], data['gamemode'])
        joc.grid = data['grid']
        joc.player = data['player']
        joc.points_x = data['points_x']
        joc.points_0 = data['points_0']
        joc.player_x_moves = data['player_x_moves']
        joc.player_0_moves = data['player_0_moves']
        # joc.moves = data['moves']
        joc.last_moves = data['last_moves']
        # joc.last_ai_move = data['last_ai_move']
        joc.points_position = data['points_position']
        joc.game_ended = data['game_ended']
        joc.winner = data['winner']
        joc.start_time = data['start_time']
        joc.time_since_last_move = data['time_since_last_move']
        joc.player_move_times = data['player_move_times']
        joc.ai_move_times = data['ai_move_times']
        joc.end_time = data['end_time']
        joc.nodes_generated = data['nodes_generated']
        joc.min_nodes_generated = data['min_nodes_generated']
        joc.max_nodes_generated = data['max_nodes_generated']
        joc.mean_nodes_generated = data['mean_nodes_generated']
        joc.median_nodes_generated = data['median_nodes_generated']

        joc.moves = joc.generate_moves(joc.player)
        return joc

    @staticmethod
    def count(grid: List[List[int]]) -> Tuple[int, int]:
        """ Functie care calculeaza punctajul fiecarui jucator al unei table de joc

        :param grid: Tabla jocului X si 0
        :return: Punctajul lui X, punctajul lui 0
        """
        points = ['eroare', 0, 0]  # points[1] = punctajul lui X, points[2] = punctajul lui 0
        n = len(grid)

        """ Parcurgem interiorul tablei """
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                # Verificam daca celula este ocupata
                if grid[i][j]:
                    # Secventa pe orizontala
                    if grid[i][j] == grid[i][j - 1] == grid[i][j + 1]:
                        points[grid[i][j]] += 1
                    # Secventa pe verticala
                    if grid[i][j] == grid[i - 1][j] == grid[i + 1][j]:
                        points[grid[i][j]] += 1
                    # Secventa pe diagonala principala
                    if grid[i][j] == grid[i - 1][j - 1] == grid[i + 1][j + 1]:
                        points[grid[i][j]] += 1
                    # Secventa pe diagonala secundara
                    if grid[i][j] == grid[i - 1][j + 1] == grid[i + 1][j - 1]:
                        points[grid[i][j]] += 1

        """ Parcurgem marginile tablei """
        for i in range(1, n - 1):
            # Prima linie
            if grid[0][i] and grid[0][i] == grid[0][i - 1] == grid[0][i + 1]:
                points[grid[0][i]] += 1
            # Ultima linie
            if grid[n - 1][i] and grid[n - 1][i] == grid[n - 1][i - 1] == grid[n - 1][i + 1]:
                points[grid[n - 1][i]] += 1

            # Prima coloana
            if grid[i][0] and grid[i][0] == grid[i - 1][0] == grid[i + 1][0]:
                points[grid[i][0]] += 1
            # Ultima coloana
            if grid[i][n - 1] and grid[i][n - 1] == grid[i - 1][n - 1] == grid[i + 1][n - 1]:
                points[grid[i][n - 1]] += 1

        return points[1], points[2]

    def count_points(self):
        """ Functie care updateaza scorul jocului curent. """
        self.points_x, self.points_0 = self.count(self.grid)

    def generate_moves(self, player):
        """ Functie care genereaza toate mutarile posibile unui jucator.

        :param player: Jucatorul care trebuie sa mute
        :return: Lista de mutari posibile
        """
        moves = []
        first_move = True
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i][j] == player:
                    first_move = False
                    moves.clear()
                    break
                elif self.grid[i][j] == 0:
                    moves.append((i, j))
            if not first_move:
                break

        if first_move:
            return moves

        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i][j] == 0:
                    flag = False
                    for x in range(max(0, i - self.K1), min(self.N, i + self.K1 + 1)):
                        for y in range(max(0, j - self.K2), min(self.N, j + self.K2 + 1)):
                            if self.grid[x][y] == player:
                                moves.append((i, j))
                                flag = True
                                break
                        if flag:
                            break
        return list(moves)

    def get_player(self):
        return "X" if self.player == 1 else "0"

    def get_opponent(self, player):
        return 3 - player

    def move_player(self, x, y):
        # Facem mutarea

        # La Load Game nu gaseste mutarea in lista de mutari posibile
        # De asta trebuie regenerate la initializare
        if (x, y) in self.moves:
            print(f"Move: ({x}, {y}) Player: {self.get_player()}")
            self.player_move_times.append(round(time.time() - self.time_since_last_move, 2))
            print(f"Time spend: {self.player_move_times[-1]} seconds")
            self.time_since_last_move = time.time()

            if self.player == 1:
                self.player_x_moves += 1
            else:
                self.player_0_moves += 1

            self.grid[x][y] = self.player
            self.last_moves.append((x, y))
            self.player = 3 - self.player
            print(self)

            if self.is_game_over():  # Verificam daca jocul s-a terminat
                self.game_over()
                return
            self.count_points()  # Actualizam scorul

            # Verificam daca jucatorul urmator are mutari posibile
            self.moves = self.generate_moves(self.player)
            if not self.moves:
                print(f"Player {self.get_player()} has no legal moves!")
                self.player = 3 - self.player
                self.moves = self.generate_moves(self.player)
                if not self.moves:
                    self.game_over()
            else:
                print(f"Player {self.get_player()} turn!")

    def undo_player(self):
        if self.last_moves:
            x, y = self.last_moves.pop()
            self.grid[x][y] = 0
            self.player = 3 - self.player
            self.moves = self.generate_moves(self.player)
            print(self)
            self.game_ended = False

    def undo_ai_player(self):
        if len(self.last_moves) > 1:
            self.undo_player()
            self.undo_player()
        elif len(self.last_moves) == 1:
            self.undo_player()
            self.player = 3 - self.player

    def make_move_ai(self, x, y):
        self.grid[x][y] = self.player
        self.player = 3 - self.player
        self.last_moves.append((x, y))

    def undo_move_ai(self, x, y):
        self.grid[x][y] = 0
        self.player = 3 - self.player
        self.last_moves.pop()

    def ai_move(self, is_maximizing, depth, alphabeta=True):
        stare = State(self, depth=depth)
        t1 = time.time()

        self.ai_moves = best_move_ai(stare, is_maximizing, alphabeta)
        self.ai_moves.sort(key=lambda x: x[0], reverse=is_maximizing)
        self.ai_move_times.append(time.time() - t1)
        self.nodes_generated.append(stare.number_nodes_generated)
        estimare, move = self.ai_moves[0]

        if move is not None:
            if self.player == 1:
                self.player_x_moves += 1
            else:
                self.player_0_moves += 1
            self.make_move_ai(*move)
            self.last_ai_move = move
            self.ai_moves.append(self.ai_moves.pop(0))
            print(self)
            print(f'Move: {move}  Estimare:{estimare}')
            print(f"AI spent {round(self.ai_move_times[-1], 2)} seconds on the move")

            if self.is_game_over():  # Verificam daca jocul s-a terminat
                self.game_over()
                return

            self.count_points()
            self.moves = self.generate_moves(self.player)
            print(f"Player {self.get_player()} turn!")
            if not self.moves:
                print(f"Player {self.get_player()} has no legal moves!")
                self.player = 3 - self.player
                self.moves = self.generate_moves(self.player)
                if not self.moves:
                    self.game_over()

        else:
            print(f"Player {self.get_player()} has no legal moves!")
            self.player = 3 - self.player

    def ai_change_move(self):
        if self.last_ai_move is not None and self.ai_moves:
            x, y = self.last_ai_move
            self.grid[x][y] = 0
            estimare, move = self.ai_moves.pop(0)
            x, y = move
            self.grid[x][y] = 3 - self.player
            self.ai_moves.append((estimare, (x, y)))

            self.last_ai_move = (x, y)
            self.ai_moves.append((estimare, move))
            self.moves = self.generate_moves(self.player)

    def ai_move_stare(self, depth):
        self.stare = self.stare.cauta_stare_dupa_mutare(self.grid)  # Mutarea jucatorului
        if depth > 3:
            t1 = time.time()
            # Calculam cea mai buna mutare din nodul celei mai bune mutari
            stare_actualizata = alpha_beta_stare(self.stare.stare_aleasa, float('-inf'), float('inf'))

        else:
            self.stare = Stare(self, self.player, depth=depth)
            t1 = time.time()

            stare_actualizata = alpha_beta_stare(self.stare, float('-inf'), float('inf'))
        self.grid = stare_actualizata.joc.grid
        self.stare = stare_actualizata
        self.ai_move_times.append(time.time() - t1)

    def game_over(self):
        self.count_points()
        if self.points_x == self.points_0:
            print("Game over! It's a tie!")
        else:
            self.winner = "X" if self.points_x > self.points_0 else "0"
            print(f"Game over! Player {self.winner} wins!")
        print(f"Player X has {self.points_x} points!")
        print(f"Player 0 has {self.points_0}  points!")
        print()
        self.end_time = round(time.time() - self.start_time, 2)
        print(f"Time elapsed: {self.end_time} seconds.")
        print(f"Number of moves made by player X: {self.player_x_moves}")
        print(f"Number of moves made by player 0: {self.player_0_moves}")

        if self.gamemode != 0:
            print(f"Time spent by AI: {round(sum(self.ai_move_times), 2)} seconds.")
            print(f"Maximum time spent by AI: {round(max(self.ai_move_times), 2)} seconds.")
            print(f"Minimum time spent by AI: {round(min(self.ai_move_times), 2)} seconds.")
            print(f"Average time spent by AI: {round(sum(self.ai_move_times) / len(self.ai_move_times), 2)} seconds.")
            print(f"Median time spent by AI: {round(statistics.median(self.ai_move_times), 2)} seconds.")

            print(f"Number of nodes generated: {sum(self.nodes_generated)}")
            print(f"Maximum number of nodes generated: {max(self.nodes_generated)}")
            print(f"Minimum number of nodes generated: {min(self.nodes_generated)}")
            print(f"Average number of nodes generated: {round(sum(self.nodes_generated) / len(self.nodes_generated), 2)}")
            print(f"Median number of nodes generated: {statistics.median(self.nodes_generated)}")

        self.game_ended = True
        self.moves.clear()

    def is_game_over(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i][j] == 0:
                    return False

        mutari_x = self.generate_moves(self.X)
        mutari_0 = self.generate_moves(self.O)
        if not mutari_x and not mutari_0:
            return True
        return True

    def in_bounds(self, x, y):
        return 0 <= x < self.N and 0 <= y < self.N

    def to_grid(self, x, y):
        return self.grid[x][y]

    def move_is_point(self, x, y):

        middle_x = [-1, -1, 0, 1, 1, 1, 0, -1]  # sens ceasornic de la 12
        middle_y = [0, 1, 1, 1, 0, -1, -1, -1]

        for k in range(4):
            coord1 = (x + middle_x[k], y + middle_y[k])
            coord2 = (x + middle_x[k + 4], y + middle_y[k + 4])
            if self.in_bounds(*coord1) and self.in_bounds(*coord2):
                if self.to_grid(*coord1) == self.to_grid(*coord2) == self.grid[x][y]:
                    return True

        edge_x = [-2, -2, 0, 2, 2, 2, 0, -2]
        edge_y = [0, 2, 2, 2, 0, -2, -2, -2]

        for k in range(8):
            coord1 = (x + edge_x[k], y + edge_y[k])
            coord2 = (x + middle_x[k], y + middle_y[k])
            if self.in_bounds(*coord1) and self.in_bounds(*coord2):
                if self.to_grid(*coord1) == self.to_grid(*coord2) == self.grid[x][y]:
                    return True

        return False

    def __str__(self):
        return "\n".join(
            " ".join(str(self.grid[i][j]).replace("0", "_").replace("1", "X").replace("2", "0") for j in range(self.N))
            for i in range(self.N))

    def __repr__(self):
        return "\n".join(" ".join(str(self.grid[i][j]) for j in range(self.N)) for i in range(self.N))


"""
#
#
#
#
#
#
#
#
#
#
"""


def estimare_pozitie_calcul():
    estimare = []

    middle_x = [-1, -1, 0, 1, 1, 1, 0, -1]  # sens ceasornic de la 12
    middle_y = [0, 1, 1, 1, 0, -1, -1, -1]
    edge_x = [-2, -2, 0, 2, 2, 2, 0, -2]
    edge_y = [0, 2, 2, 2, 0, -2, -2, -2]

    def in_bounds(n, i, j):
        return 0 <= i < n and 0 <= j < n

    for N in range(0, 11, 1):
        matrix = np.zeros((N, N), dtype=int)
        for x in range(N):
            for y in range(N):
                points = 0
                for k in range(4):
                    coord1 = (x + middle_x[k], y + middle_y[k])
                    coord2 = (x + middle_x[k + 4], y + middle_y[k + 4])
                    if in_bounds(N, *coord1) and in_bounds(N, *coord2):
                        points += 1

                for k in range(8):
                    coord1 = (x + edge_x[k], y + edge_y[k])
                    coord2 = (x + middle_x[k], y + middle_y[k])
                    if in_bounds(N, *coord1) and in_bounds(N, *coord2):
                        points += 1
                if x == y and points == 12:
                    points *= 2

                matrix[x][y] = points

        estimare.append(matrix)

    return estimare


estimare_pozitie = estimare_pozitie_calcul()


class State:

    def __init__(self, joc, depth):
        self.joc = copy.deepcopy(joc)
        self.depth = depth

        self.number_nodes_generated = 0

    def __dict__(self):
        return self.joc.__dict__

    def estimare1(self):
        grid = self.joc.grid
        n = self.joc.N
        scor = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == self.joc.player:
                    scor += estimare_pozitie[n][i][j]
                elif grid[i][j] == 3 - self.joc.player:
                    scor -= estimare_pozitie[n][i][j]

        return scor

    def estimare2(self):
        grid = self.joc.grid
        n = self.joc.N

        middle_x = [-1, -1, 0, 1, 1, 1, 0, -1]
        middle_y = [0, 1, 1, 1, 0, -1, -1, -1]
        edge_x = [-2, -2, 0, 2, 2, 2, 0, -2]
        edge_y = [0, 2, 2, 2, 0, -2, -2, -2]

        def in_bounds(n, i, j):
            return 0 <= i < n and 0 <= j < n

        def to_grid(i, j):
            return grid[i][j]

        estimare = 0
        for x in range(n):
            for y in range(n):
                if grid[x][y] != self.joc.player:
                    continue
                for k in range(4):
                    coord1 = (x + middle_x[k], y + middle_y[k])
                    coord2 = (x + middle_x[k + 4], y + middle_y[k + 4])
                    if in_bounds(n, *coord1) and in_bounds(n, *coord2):
                        if to_grid(*coord1) == to_grid(*coord2) == 0:
                            estimare += 1
                        elif to_grid(*coord1) == 0 and to_grid(*coord2) == self.joc.player \
                                or to_grid(*coord1) == self.joc.player and to_grid(*coord2) == 0:
                            estimare += 2

                for k in range(8):
                    coord1 = (x + edge_x[k], y + edge_y[k])
                    coord2 = (x + middle_x[k], y + middle_y[k])
                    if in_bounds(n, *coord1) and in_bounds(n, *coord2):
                        if to_grid(*coord1) == to_grid(*coord2) == 0:
                            estimare += 1
                        elif to_grid(*coord1) == 0 and to_grid(*coord2) == self.joc.player \
                                or to_grid(*coord1) == self.joc.player and to_grid(*coord2) == 0:
                            estimare += 2

        return estimare

    def estimare(self):
        return self.estimare2() * 10 + self.estimare1() * 10
        # return self.estimare1() * 10
        # return self.estimare2() * 10

    def __str__(self) -> str:
        return str(self.joc)


def min_max(stare, depth, is_maximizing):
    if depth == 0 or stare.joc.is_game_over():
        scor_x, scor_0 = TicTacToe.count(stare.joc.grid)

        if is_maximizing:
            return (scor_x - scor_0) * 1000 + stare.estimare()
        else:
            return (scor_x - scor_0) * 1000 - stare.estimare()

    elif is_maximizing:
        best_score = float('-inf')
        possible_moves = stare.joc.generate_moves(stare.joc.player)
        stare.number_nodes_generated += len(possible_moves)

        for move in possible_moves:
            stare.joc.make_move_ai(*move)
            score = min_max(stare, depth - 1, False)
            stare.joc.undo_move_ai(*move)
            best_score = max(score, best_score)
        return best_score
    elif not is_maximizing:
        best_score = float('inf')
        possible_moves = stare.joc.generate_moves(stare.joc.player)
        stare.number_nodes_generated += len(possible_moves)

        for move in possible_moves:
            stare.joc.make_move_ai(*move)
            score = min_max(stare, depth - 1, True)
            stare.joc.undo_move_ai(*move)
            best_score = min(score, best_score)
        return best_score


def alpha_beta(stare, depth, alpha, beta, is_maximizing):
    if depth == 0 or stare.joc.is_game_over():  # aici trebuie implementat si pt cazul in care jocul s-a terminat dar mai are positii libere
        scor_x, scor_0 = TicTacToe.count(stare.joc.grid)

        if is_maximizing:
            return (scor_x - scor_0) * 100 + stare.estimare()
        else:
            return (scor_x - scor_0) * 100 - stare.estimare()

    elif is_maximizing:
        best_score = float('-inf')
        possible_moves = stare.joc.generate_moves(stare.joc.player)
        stare.number_nodes_generated += len(possible_moves)

        for move in possible_moves:
            stare.joc.make_move_ai(*move)
            score = alpha_beta(stare, depth - 1, alpha, beta, False)
            stare.joc.undo_move_ai(*move)

            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score

    elif not is_maximizing:
        best_score = float('inf')
        possible_moves = stare.joc.generate_moves(stare.joc.player)
        stare.number_nodes_generated += len(possible_moves)

        for move in possible_moves:
            stare.joc.make_move_ai(*move)
            score = alpha_beta(stare, depth - 1, alpha, beta, True)
            stare.joc.undo_move_ai(*move)

            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score


def best_move_ai(stare, is_maximizing, alpha_beta_flag):
    ai_moves = []
    possible_moves = stare.joc.generate_moves(stare.joc.player)
    stare.number_nodes_generated += len(possible_moves)

    for move in possible_moves:
        stare.joc.make_move_ai(*move)
        if alpha_beta_flag:
            score = alpha_beta(stare, stare.depth - 1, float('-inf'), float('inf'), not is_maximizing)
        else:
            score = min_max(stare, stare.depth - 1, not is_maximizing)
        stare.joc.undo_move_ai(*move)
        ai_moves.append((score, move))

    return ai_moves


"""
#
#
#
#
#
#
#
#
#
#
#
"""


class Stare:
    def __init__(self, joc, player, depth, estimare=None):
        self.joc = joc  # folosit strict pentru grid, k1, k2
        self.player = player
        self.depth = depth
        self.estimare = estimare
        self.mutari_posibile = []
        self.alphabeta = {}


        # cea mai bună mutare din lista de mutari posibile pentru jucatorul curent
        # e de tip Stare (cel mai bun succesor)
        self.stare_aleasa = None

        self.number_nodes_generated = 0

    def cauta_stare_dupa_mutare(self, grid):
        for stare in self.mutari_posibile:
            if stare.joc.grid == grid:
                return stare


    def estimare1(self):
        grid = self.joc.grid
        n = self.joc.N
        scor = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] == self.joc.player:
                    scor += estimare_pozitie[n][i][j]
                elif grid[i][j] == 3 - self.joc.player:
                    scor -= estimare_pozitie[n][i][j]

        return scor

    def estimare2(self):
        grid = self.joc.grid
        n = self.joc.N

        middle_x = [-1, -1, 0, 1, 1, 1, 0, -1]
        middle_y = [0, 1, 1, 1, 0, -1, -1, -1]
        edge_x = [-2, -2, 0, 2, 2, 2, 0, -2]
        edge_y = [0, 2, 2, 2, 0, -2, -2, -2]

        def in_bounds(N, i, j):
            return 0 <= i < N and 0 <= j < N

        def to_grid(i, j):
            return grid[i][j]

        estimare = 0
        for x in range(n):
            for y in range(n):
                if grid[x][y] != self.joc.player:
                    continue
                for k in range(4):
                    coord1 = (x + middle_x[k], y + middle_y[k])
                    coord2 = (x + middle_x[k + 4], y + middle_y[k + 4])
                    if in_bounds(n, *coord1) and in_bounds(n, *coord2):
                        if to_grid(*coord1) == to_grid(*coord2) == 0:
                            estimare += 1
                        elif to_grid(*coord1) == 0 and to_grid(*coord2) == self.joc.player \
                                or to_grid(*coord1) == self.joc.player and to_grid(*coord2) == 0:
                            estimare += 2

                for k in range(8):
                    coord1 = (x + edge_x[k], y + edge_y[k])
                    coord2 = (x + middle_x[k], y + middle_y[k])
                    if in_bounds(n, *coord1) and in_bounds(n, *coord2):
                        if to_grid(*coord1) == to_grid(*coord2) == 0:
                            estimare += 1
                        elif to_grid(*coord1) == 0 and to_grid(*coord2) == self.joc.player \
                                or to_grid(*coord1) == self.joc.player and to_grid(*coord2) == 0:
                            estimare += 2

        return estimare

    def estimare3(self):
        points_x, points_y = TicTacToe.count(self.joc.grid)
        return (points_x - points_y)

    def estimare4(self):
        return self.estimare3() * 100 + self.estimare2() * 10 + self.estimare1()

    def estimare_calcul(self):
        return self.estimare4()


    def mutari(self):
        l_mutari = []
        mutari_posibile = self.joc.generate_moves(self.player)

        self.number_nodes_generated += len(mutari_posibile)
        for move in mutari_posibile:
            joc_nou = copy.deepcopy(self.joc)
            joc_nou.make_move_ai(*move)
            l_mutari.append(Stare(joc_nou, 3 - self.player, self.depth - 1, self.estimare))
        return l_mutari

    def __str__(self):
        return f"{self.joc}\nPlayer: {self.player}"

    def __repr__(self):
        return str(self)



def min_max_stare(stare):
    if stare.depth == 0 or stare.joc.is_game_over():
        stare.estimare = stare.estimare_calcul()
        return stare

    stare.mutari_posibile = stare.mutari()
    mutari_cu_estimare = [min_max_stare(x) for x in stare.mutari_posibile]

    if stare.player == 1:
        stare.stare_aleasa = max(mutari_cu_estimare, key=lambda x: x.estimare)
    else:
        stare.stare_aleasa = min(mutari_cu_estimare, key=lambda x: x.estimare)

    stare.estimare = stare.stare_aleasa.estimare
    return stare


def alpha_beta_stare(stare, alpha, beta):
    if stare.depth == 0 or stare.joc.is_game_over():
        stare.estimare = stare.estimare_calcul()
        return stare

    if alpha > beta:
        return stare

    stare.mutari_posibile = stare.mutari()
    if stare.player == 1:
        estimare_curenta = float('-inf')
        if stare.mutari_posibile is not None:
            stare.mutari_posibile.sort(key=lambda x: x.estimare(), reverse=True)    # Sortam mutarile in functie de estimare inainte de expandare
        for mutare in stare.mutari_posibile:
            stare_noua = alpha_beta_stare(mutare, alpha, beta)  # aici construim subarborele pentru stare_noua
            if (estimare_curenta < stare_noua.estimare):
                stare.stare_aleasa = stare_noua
                estimare_curenta = stare_noua.estimare

            if (alpha < stare_noua.estimare):
                alpha = stare_noua.estimare
                if alpha >= beta:  # interval invalid
                    break

    elif stare.player == 2:
        estimare_curenta = float('inf')
        if stare.mutari_posibile is not None:
            stare.mutari_posibile.sort(key=lambda x: x.estimare(), reverse=False)   # Sortam mutarile in functie de estimare
        for mutare in stare.mutari_posibile:
            stare_noua = alpha_beta_stare(mutare, alpha, beta)
            if (estimare_curenta > stare_noua.estimare):
                stare.stare_aleasa = stare_noua
                estimare_curenta = stare_noua.estimare

            if (beta > stare_noua.estimare):
                beta = stare_noua.estimare
                if alpha >= beta:
                    break

    stare.estimare = stare.stare_aleasa.estimare
    return stare