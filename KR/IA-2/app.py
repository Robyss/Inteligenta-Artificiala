from game_state import GameState

""" Joc X si 0 pe steroizi.
    Grid N x N, cu 4 <= N <= 10
    Restrictiile K1 si K2, cu 1 < K1 < K2 < N, unde:
    - K1 reprezinta distanta maxima de linii dintre viitoarea mutare și una din mutarile jucătorului
    - K2 reprezinta distanta maxima de coloane dintre viitoarea mutare și una din mutarile jucătorului
"""

if __name__ == "__main__":
    game = GameState()
    game.run()