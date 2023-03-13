"""
Problema blocurilor

Blocuri de intervale [a,b]
O mutare se poate efectua doar dacă intervalele celor 2 blocuri (cel mutat și baza) se intersecteaza.
Nu este permisă mutarea unui bloc pe un alt bloc cu același interval).
Scopul este ca fiecare bloc să se afle în intervalul blocului de sub el.
Costul unei mutări este b - a.

"""

from __future__ import annotations
from collections import deque
from typing import Optional, List
from stopit import threading_timeoutable
import heapq
import os
import sys
import copy
import re
import time


class Stack(deque):
    """
        Clasa derivata din structura de date Deque ce simuleaza o stiva.
        Metodele detinute de aceasta clasa sunt:
            - push(x):  Adauga in stiva elementul x
            - top():    Returneaza primul element din stiva
            - pop():    Returneaza primul element din stiva pe care il si elimina
    """

    def push(self, x):
        """ Functia append() redenumita in push(). """
        self.append(x)

    def top(self):
        """ Returneaza elementul deasupra stivei. """
        if len(self) == 0:
            raise IndexError("Eroare top()! Nu se afla niciun element în Stiva!")
        return self[-1]

    def pop(self):
        """ Elimina elementul deasupra stivei pe care il returneaza. """
        return super().pop()


class Cub:
    """
    Clasa Cub cuprinde informatii si metode asupra unui bloc format dintr-un interval din stiva.

    """

    def __init__(self, a: int, b: int) -> None:
        """ Constructorul clasei Cub.

        :param a: Intreg ce reprezinta marginea stanga a intervalului
        :param b: Intreg ce reprezinta marginea dreapta a intervalului
        """

        assert a <= b, "Intervalul cubului invalid"

        self.st = a
        self.dr = b

    def __len__(self):
        return self.dr - self.st

    # Comparare cub1 == cub2
    def __eq__(self, cub) -> bool:
        return self.st == cub.st and self.dr == cub.dr

    def __hash__(self):
        return hash((self.st, self.dr))

    # Acelasi lucru cu len(self)
    def cost_mutare(self) -> int:
        return len(self)

    def mutare_valida(self, cub):
        # Daca cuburile au acelasi interval
        if self == cub:
            return False
        # Daca cuburile se intersecteaza
        if self.st <= cub.dr and cub.st <= self.dr:
            return True
        return False

    def in_interval(self, cub) -> bool:
        # Daca cubul self se afla in intervalul cubului cub
        return cub.st <= self.st and cub.dr >= self.dr

    def __str__(self) -> str:
        return str(f'[{self.st}, {self.dr}]')

    def __repr__(self) -> str:
        return str(self)


class Nod:
    """
        Clasa Nod este folosita pentru a stoca o lista de stive, impreuna cu referinta
        din care provine nodul curent, costul mutarii si euristica.
        O stiva contine obiecte de tip Cub ce reprezinta intervalele problemei.
    """

    def __init__(self, stive: List[Stack], parinte: Optional[Nod], cost: int, h: int) -> None:
        """
        Constructorul clasei Nod.

        :param stive:   Lista de stive a unei stari.
        :param parinte: Obiect de tip Nod ce reprezinta parintele nodului curent.
                        Este None pentru radacina grafului.
        :param cost:    Costul de a ajunge din nodul parinte in nodul curent.
        :param h:       Valoarea euristicii
        """
        self.stive = stive
        self.parinte = parinte
        self.cost = cost
        self.h = h
        self.g = 0
        if parinte is not None:
            self.g = parinte.g + cost

    def obtine_drum(self) -> List[Nod]:
        """ Functie ce formeaza drumul de la radacina la nodul self """
        drum = []
        nod = self
        while nod is not None:
            drum.append(nod)
            nod = nod.parinte
        return list(reversed(drum))

    def afisare_drum(self, afis_cost: bool = False, file=sys.stdout) -> None:
        """ Functia de afisare a solutiilor. Foloseste functia obtine_drum() pentru a construi drumul. """
        drum = self.obtine_drum()  # Lista de noduri
        for i, nod in enumerate(drum):
            print(f'{i})\n{nod}', file=file)
            if afis_cost:
                print(f'Costul drumului: {nod.g}     Costul mutarii: {nod.cost}', file=file)
            print(file=file)
        print(f'Drum de lungime: {len(drum) - 1}', file=file)
        if afis_cost:
            print(f'Costul total: {drum[len(drum) - 1].g}', file=file)

    def contine_in_drum(self, nod: Nod) -> bool:
        """ Functie care afla daca o stare a unui nod este deja existenta in drumul curent. """
        nod_curent = self
        while nod_curent is not None:
            if nod == nod_curent:
                return True
            nod_curent = nod_curent.parinte
        return False

    def testeaza_scop(self) -> bool:
        """ Testam nodul daca este solutie. """
        for stack in self.stive:
            for i in range(len(stack) - 1, 0, -1):  # Parcurgem stiva de sus in jos
                if not stack[i].in_interval(stack[i - 1]):
                    return False
        return True

    def genereaza_succesori(self, euristica: str = "") -> List[Nod]:
        """
            Functia de generare a nodurilor.

            Pentru fiecare element varf din stiva identificam
            daca exista o posibila mutare pe alta stiva.
            Returneaza o lista de noduri.
        """
        succesori = []
        for i, stack1 in enumerate(self.stive):
            for j, stack2 in enumerate(self.stive):
                if i == j or len(stack1) == 0:
                    continue  # Ignoram mutarea pe aceeasi stiva/Nu exista mutari de pe o stiva vida
                cub = stack1.top()  # Valoarea cubul ce trebuie mutat

                if len(stack2) == 0 or cub.mutare_valida(stack2.top()):
                    stive_copie = copy.deepcopy(self.stive)  # Copiem stiva intr-o copie
                    stive_copie[j].push(stive_copie[i].pop())  # Adaugam cubul din stiva veche in noua stiva

                    # Cream noul nod in zona de memorie
                    nod_nou = Nod(stive=stive_copie, parinte=self, cost=cub.cost_mutare(), h=1)
                    if euristica:  # daca avem cautare informata, calculam euristica
                        nod_nou.h = self.calculeaza_h(nod=nod_nou, tip_euristica=euristica)

                    if not self.contine_in_drum(nod_nou):  # Daca nodul nu se afla in arbore
                        succesori.append(nod_nou)  # il adaugam in lista de succesori
        return succesori

    @staticmethod
    def calculeaza_h(nod: Nod, tip_euristica: str) -> int:
        if tip_euristica == "banala":
            if nod.testeaza_scop():
                return 0
            return 1

        if tip_euristica == "admisibila 1":
            stive = nod.stive
            cost = 0
            for stiva in stive:
                for i in range(len(stiva) - 1):
                    if not stiva[i + 1].in_interval(stiva[i]):
                        cost += 1
                        break
            return cost

        if tip_euristica == "admisibila 2":
            stive = nod.stive
            cost = 0
            for stiva in stive:
                for i in range(len(stiva) - 1):
                    if not stiva[i + 1].in_interval(stiva[i]):
                        cost += len(stiva) - i - 1
                        break
            return cost

        if tip_euristica == "sef":
            stive = nod.stive
            cost = 0
            for stiva in stive:
                for i in range(len(stiva) - 1):
                    if not stiva[i + 1].in_interval(stiva[i]):
                        for j in range(i + 1, len(stiva)):
                            cost += stiva[j].cost_mutare()
                        break
            return cost
        if tip_euristica == "neadmisibila":
            stive = nod.stive
            cost = 0
            cost_max = []
            for stiva in stive:
                if len(stiva):
                    cost_max.append(max(len(cub) for cub in stiva))
            cost_max = max(cost_max)

            for stiva in stive:
                for i in range(len(stiva) - 1):
                    if not stiva[i + 1].in_interval(stiva[i]):
                        cost += len(stiva) - i - 1
                        break
            return cost * cost_max

    def exista_solutie(self) -> bool:
        """ Testam daca exista solutii pentru o stare initiala. """

        # Testam daca exista o mutare valida
        solutii = self.genereaza_succesori()
        if len(solutii) == 0:
            return False

        # # Testam daca numarul de intervale este mai mic decat numarul de stive
        intervale = [self.stive[0][0]]
        for stiva in self.stive:  # Parcurgem stivele
            for cub in stiva:
                found = False
                for interval in intervale:
                    if cub.in_interval(interval):
                        found = True
                        break
                    elif interval.in_interval(cub):
                        found = True
                        intervale.remove(interval)
                        intervale.append(cub)
                        break
                if not found:
                    intervale.append(cub)

        return len(intervale) <= len(self.stive)

    def __eq__(self, o: Nod) -> bool:
        """
        Functio de comparare obiect1 == obiect2
        Returneaza True daca cuburile stivelor obiectului 1
        au aceleasi valori cu cuburile stivelor obiectului 2,
        respectiv False in caz contrar.

        :param o: Obiect de tip Nod
        :return: True/False
        """
        assert isinstance(o, Nod), "Comparare structura de date eronata!"

        for stack1, stack2 in zip(self.stive, o.stive):
            if len(stack1) != len(stack2):
                return False
            for cub1, cub2 in zip(stack1, stack2):
                if cub1 != cub2:
                    return False
        return True

    def __lt__(self, other):
        """ Trebuie creat operatorul < pentru priority_queue.
        Folosit in special pentru A* la ordonarea in functie de f(nod)
        """
        if self.h + self.g < other.h + other.g:  # Daca f(nod1) e mai mic decat f(nod2)
            return True

        if self.h + self.g == other.h + other.g:  # Daca f-urile sunt egale, mai apropriat
            return self.h < other.h  # de raspuns este nodul cu h-ul mai mic

        return False

    def __hash__(self):
        """ Hashcode pentru set. """
        return hash(tuple(tuple(cub) for cub in self.stive))

    def __str__(self) -> str:
        """ Afisarea pe vertical a stivelor. """
        sir = ""
        max_inalt = max([len(stiva) for stiva in self.stive])
        for inalt in range(max_inalt, 0, -1):
            for stiva in self.stive:
                if len(stiva) < inalt:
                    sir += "\t\t"
                else:
                    sir += str(stiva[inalt - 1]) + "  "
            sir += "\n"
        sir += "----" * (2 * len(self.stive))
        return sir

    def __repr__(self) -> str:
        rezultat = ""
        for stiva in self.stive:
            for cub in stiva:
                rezultat += str(cub)
            rezultat += '\n'
        return rezultat


def solution_guard(method):
    """
    Metoda de a preveni apelarea functiilor de rezolvare
    a unui graf pentru un nod initial scop.
    """

    def wrapper(self, *args, **kwargs):
        if self.radacina.testeaza_scop():
            return "Starea initiala este deja o solutie"
        if not self.radacina.exista_solutie():
            return "Nu exista solutii pentru starea initiala"
        return method(self, *args, **kwargs)

    return wrapper


def get_stacks(lines: List[str]) -> List[Stack]:
    """ Genereaza lista de stive din fisier.

    Programul ignora orice caracter in plus, insa pastreaza
    continutul intervalului de forma [a,b] printr-un regex.
    De asemenea, 0 singur pe rand reprezinta o stiva libera.
    """
    rezultat = []

    for line in lines:
        stiva_curenta = Stack()
        intervale = re.findall(r'\[(.*?)]', line)

        for interval in intervale:
            a, b = map(int, interval.split(','))  # De modificat aici pentru intervale cu numere reale
            stiva_curenta.push(Cub(a, b))

        if intervale or (line == "0\n" or line == "0"):
            rezultat.append(stiva_curenta)
    return rezultat


def read_from_file(file_path: str) -> List[Stack]:
    """ Functie ce citeste dintr-un fisier si returneaza o lista de stive. """
    with open(file_path) as file:
        lines = file.readlines()
    return get_stacks(lines)


class Graf:
    """
    Clasa Graf detine metodele de cautare ale grafului, precum:
    - BFS, DFS, DFI, UCS, Greedy, A*, IDA*     => Genereaza NSOL drumuri.
    - A* optimizat      => Genereaza drumul de cost minim.

    Variabilele unei instante sunt urmatoarele:

    - radacina: Nod     = Reprezinta starea initiala a problemei.
    - solutii: List[Tuple(Nod, time, int, int)]  =>
        - Nod   = Nodul scop al solutiei respective
        - time  = Timpul de generare al solutiei
        - int   = Numarul maxim de noduri existente in memorie la momentul respectiv
        - int   = Numarul total de noduri generate
    - nsol: int          = Numarul de solutii dorite
    - timeout: int       = Numarul de secunde pana la timeout
    - fisier_input: str  = Numele fisierului de input
    - folder_output: str = Numele folderului de output
    """

    euristica = ["banala", "admisibila 1", "admisibila 2", "sef", "neadmisibila"]
    metoda = ["BFS", "DFS", "DFI", "UCS", "Greedy", "A_star", "A_star_optim", "IDA_star"]

    def __init__(self, fisier_input: str, folder_output: str = None, nsol: int = 1, timeout: int = 0) -> None:

        # Initializam radacina cu elementele din fisier
        self.radacina = Nod(stive=read_from_file(fisier_input), parinte=None, cost=0, h=1)
        self.solutii = []
        self.fisier_input = fisier_input.split("\\")[-1]
        self.folder_output = folder_output
        self.nsol = nsol
        self.timeout = timeout

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    def afisare_solutii(self, metoda: str, euristica: str = "", afis_cost=True):

        if euristica:
            cale = os.path.join(self.folder_output, metoda, euristica, self.fisier_input.replace(".in", ".out"))
        else:
            cale = os.path.join(self.folder_output, metoda, self.fisier_input.replace(".in", ".out"))
        output = open(cale, "w")

        if len(self.solutii) == 0:
            print(f"Nu exista solutii pentru metoda {metoda}", file=output)
            self.solutii.clear()
            output.close()
            return

        print(f'Metoda {metoda}', file=output)
        print(f'Numarul de solutii generate: {len(self.solutii)}\n', file=output)

        if metoda != "A_star_optim" and self.nsol > len(self.solutii):
            print(f'Nu exista {self.nsol} solutii pentru metoda {metoda}.\n', file=output)

        for i, (nod, timp, noduri_memorie, noduri_generate) in enumerate(self.solutii):
            print(f"Solutia nr {i + 1}: ", file=output)
            nod.afisare_drum(file=output, afis_cost=afis_cost)
            print(f"Timpul de executie: {timp * 1000:.4f} ms | {timp:.4f} s", file=output)
            print(f'Numarul maxim de noduri existente in memorie: {noduri_memorie}', file=output)
            print(f'Numarul total de noduri calculate: {noduri_generate}', file=output)

            print(file=output)
            print("######" * 2 * len(self.radacina.stive), "\n", file=output)

        self.solutii.clear()
        output.close()

    @solution_guard
    @threading_timeoutable(default="Timeout la BFS")
    def bfs(self, n_sol: int = 1, optim: bool = True):
        """ Cautare neinformata de tip breath first. """
        if optim:
            self.__bfs_optim(n_sol)
        else:
            self.__bfs_lent(n_sol)
        self.afisare_solutii(self.metoda[0], afis_cost=False)

    def __bfs_lent(self, n_sol):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        coada = [self.radacina]
        while len(coada) and n_sol:
            nod = coada.pop(0)
            if nod.testeaza_scop():
                self.solutii.append((nod, time.time() - t1, noduri_in_memorie, noduri_generate))
                n_sol -= 1
            succesori = nod.genereaza_succesori()
            coada.extend(succesori)

            noduri_in_memorie = max(noduri_in_memorie, len(coada))
            noduri_generate += len(succesori)

    def __bfs_optim(self, n_sol):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        coada = [self.radacina]
        while len(coada) and n_sol:
            nod = coada.pop(0)
            succesori = nod.genereaza_succesori()
            for succesor in succesori:
                if not n_sol:
                    return
                if succesor.testeaza_scop():
                    self.solutii.append((nod, time.time() - t1, noduri_in_memorie, noduri_generate))
                    n_sol -= 1
                else:
                    coada.append(succesor)

            noduri_in_memorie = max(noduri_in_memorie, len(coada))
            noduri_generate += len(succesori)

    @solution_guard
    @threading_timeoutable(default="Timeout la DFS")
    def dfs(self, n_sol=1):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0
        self.__dfs(self.radacina, n_sol, t1, noduri_in_memorie, noduri_generate)
        self.afisare_solutii(self.metoda[1], afis_cost=False)

    def __dfs(self, nod_curent, n_sol, timp_start, noduri_in_memorie, noduri_generate):
        if not n_sol:
            return

        if nod_curent.testeaza_scop():
            self.solutii.append((nod_curent, time.time() - timp_start, noduri_in_memorie, noduri_generate))
            n_sol -= 1
            return n_sol

        succesori = nod_curent.genereaza_succesori()
        noduri_generate += len(succesori)
        for nod in succesori:
            if n_sol:
                n_sol = self.__dfs(nod, n_sol, timp_start, max(noduri_in_memorie, len(nod.obtine_drum())),
                                   noduri_generate)
            if not n_sol:
                return n_sol
        return n_sol

    @solution_guard
    @threading_timeoutable(default="Timeout la DFI")
    def dfi(self, n_sol=1):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        for adancime in range(1, 1000):
            if not n_sol:
                break
            n_sol = self.__dfi(n_sol, self.radacina, t1, adancime, noduri_in_memorie, noduri_generate)
        self.afisare_solutii(self.metoda[2])

    def __dfi(self, n_sol, nod_curent, timp_start, adancime, noduri_in_memorie, noduri_generate):
        if not n_sol:
            return

        if nod_curent.testeaza_scop():
            self.solutii.append((nod_curent, time.time() - timp_start, noduri_in_memorie, noduri_generate))
            n_sol -= 1
            return n_sol

        if adancime == 0:
            return n_sol

        succesori = nod_curent.genereaza_succesori()
        noduri_generate += len(succesori)
        for nod in succesori:
            if n_sol:
                n_sol = self.__dfi(n_sol, nod, timp_start, adancime - 1, max(noduri_in_memorie, len(nod.obtine_drum())),
                                   noduri_generate)

        return n_sol

    @solution_guard
    @threading_timeoutable(default="Timeout la UCS")
    def ucs(self, n_sol=1):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        priority_queue = []
        heapq.heappush(priority_queue, (0, self.radacina))

        while priority_queue:
            if not n_sol:
                self.afisare_solutii(self.metoda[3])
                return

            cost, nod_curent = heapq.heappop(priority_queue)

            if nod_curent.testeaza_scop():
                self.solutii.append((nod_curent, time.time() - t1, noduri_in_memorie, noduri_generate))
                n_sol -= 1
                continue

            # Adaugam in priority queue fiecare nod generat si costul drumului pana la acel nod
            succesori = nod_curent.genereaza_succesori()
            [heapq.heappush(priority_queue, (nod.g, nod)) for nod in succesori]

            noduri_in_memorie = max(noduri_in_memorie, len(priority_queue))
            noduri_generate += len(succesori)

        self.afisare_solutii(self.metoda[3])

    @solution_guard
    @threading_timeoutable(default="Timeout la Greedy")
    def greedy(self, n_sol=1, euristica="sef"):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        priority_queue = []
        heapq.heappush(priority_queue, (0, self.radacina))

        while priority_queue:
            if n_sol == 0:
                self.afisare_solutii(self.metoda[4], euristica)
                return

            h, nod_curent = heapq.heappop(priority_queue)

            if nod_curent.testeaza_scop():
                self.solutii.append((nod_curent, time.time() - t1, noduri_in_memorie, noduri_generate))
                n_sol -= 1
                continue

            # Adaugam in priority queue fiecare nod generat si costul drumului pana la acel nod
            succesori = nod_curent.genereaza_succesori(euristica)
            [heapq.heappush(priority_queue, (nod.h, nod)) for nod in nod_curent.genereaza_succesori(euristica)]

            noduri_in_memorie = max(noduri_in_memorie, len(priority_queue))
            noduri_generate += len(succesori)

        self.afisare_solutii(self.metoda[4], euristica)

    @solution_guard
    @threading_timeoutable(default="Timeout la A*")
    def a_star(self, n_sol=1, euristica="sef"):
        """ A* ce genereaza mai multe drumuri. """
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        priority_queue = []
        heapq.heappush(priority_queue, self.radacina)  # Introducem radacina in pq, avem __eq__ si __lt__

        while priority_queue:
            if not n_sol:
                self.afisare_solutii(self.metoda[5], euristica)
                return

            nod_curent = heapq.heappop(priority_queue)

            if nod_curent.h == 0:  # Daca nodul este scop
                self.solutii.append((nod_curent, time.time() - t1, noduri_in_memorie, noduri_generate))
                n_sol -= 1
            else:
                succesori = nod_curent.genereaza_succesori(euristica=euristica)
                for nod in succesori:
                    heapq.heappush(priority_queue, nod)

                noduri_in_memorie = max(noduri_in_memorie, len(priority_queue))
                noduri_generate += len(succesori)

        # print(f"Numar solutii generate: {len(self.solutii)}\n"
        #       f"Nu s-au putut genera {n_sol + len(self.solutii)} solutii", file=self.output_file)
        self.afisare_solutii(self.metoda[5], euristica)

    @solution_guard
    @threading_timeoutable(default="Timeout la A* optimizat")
    def a_star_optim(self, euristica="sef"):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        open_l = []  # priority queue
        closed_l = set()  # nodurile expandate
        heapq.heappush(open_l, self.radacina)

        while open_l:
            nod_curent = heapq.heappop(open_l)
            closed_l.add(nod_curent)
            if nod_curent.h == 0:
                self.solutii.append((nod_curent, time.time() - t1, noduri_in_memorie, noduri_generate))
                self.afisare_solutii(self.metoda[6], euristica)
                return
            else:
                succesori = nod_curent.genereaza_succesori(euristica=euristica)
                for nod in succesori:
                    if nod not in closed_l:
                        heapq.heappush(open_l, nod)

                noduri_in_memorie = max(noduri_in_memorie, len(open_l))
                noduri_generate += len(succesori)

        self.afisare_solutii(self.metoda[6], euristica)

    def __a_star_optim2(self, euristica="sef"):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        open_l = []
        heapq.heappush(open_l, self.radacina)
        closed_l = set()

        while open_l:
            nod_curent = heapq.heappop(open_l)
            closed_l.add(nod_curent)

            if nod_curent.h == 0:
                self.solutii.append((nod_curent, time.time() - t1, noduri_in_memorie, noduri_generate))
                self.afisare_solutii(self.metoda[6])
                return
            else:
                succesori = nod_curent.genereaza_succesori(euristica=euristica)
                for succesor in succesori:
                    if succesor in closed_l:
                        continue
                    nod_open = next(filter(lambda nod: nod == succesor, open_l), None)
                    if nod_open is None:
                        heapq.heappush(open_l, succesor)
                    else:
                        # if succesor.g + succesor.h < nod_open.g + nod_open.h:
                        if succesor < nod_open:  # succesor.f < nod_open.f
                            nod_open.parinte = succesor.parinte
                            nod_open.cost = succesor.cost
                            nod_open.g = succesor.g
                            nod_open.h = succesor.h
                noduri_in_memorie = max(noduri_in_memorie, len(open_l))
                noduri_generate += len(succesori)

        self.afisare_solutii(self.metoda[6])

    @solution_guard
    @threading_timeoutable(default="Timeout la IDA*")
    def ida_star(self, n_sol, euristica="sef"):
        t1 = time.time()
        noduri_in_memorie = 1
        noduri_generate = 0

        nod_start = self.radacina
        limita = nod_start.g + nod_start.h

        def __ida_star(nod_curent, limita, n_sol, noduri_in_memorie, noduri_generate):
            if nod_curent.g + nod_curent.h > limita:
                return n_sol, nod_curent.g + nod_curent.h
            if nod_curent.testeaza_scop():
                self.solutii.append((nod_curent, time.time() - t1, noduri_in_memorie, noduri_generate))
                n_sol -= 1
                if n_sol == 0:
                    return 0, "gata"
            succesori = nod_curent.genereaza_succesori(euristica=euristica)
            noduri_generate += len(succesori)

            minim = float('inf')
            for succesor in succesori:
                n_sol, rez = __ida_star(succesor, limita, n_sol, max(noduri_in_memorie, len(succesor.obtine_drum())),
                                        noduri_generate)
                if rez == "gata":
                    return 0, "gata"
                if rez < minim:
                    minim = rez
            return n_sol, minim

        while True:
            n_sol, rez = __ida_star(nod_start, limita, n_sol, noduri_in_memorie, noduri_generate)
            if rez == "gata":
                break
            if rez == float('inf'):
                break
            limita = rez
        self.afisare_solutii(self.metoda[7], euristica)

    def solve(self):
        fisier = self.fisier_input.replace(".in", ".out")
        somn = 0.1

        print(f"Testarea fisierului {fisier}"), time.sleep(somn)

        print("Metoda 1: BFS"), time.sleep(somn)
        temp = self.bfs(n_sol=self.nsol, timeout=self.timeout)
        if temp is not None:
            print(temp, file=open(os.path.join(self.folder_output, self.metoda[0], fisier), "w"))
            print(temp), time.sleep(somn)
        else:
            print("Metoda incheiata cu succes"), time.sleep(somn)

        print("Metoda 2: DFS"), time.sleep(somn)
        try:
            temp = self.dfs(n_sol=self.nsol, timeout=self.timeout)
            if temp is not None:
                print(temp, file=open(os.path.join(self.folder_output, self.metoda[1], fisier), "w"))
                print(temp), time.sleep(somn)
            else:
                print("Metoda incheiata cu succes"), time.sleep(somn)
        except Exception as e:
            print(e, file=open(os.path.join(self.folder_output, self.metoda[1], fisier), "w"))
            print(e), time.sleep(somn)

        print("Metoda 3: DFI"), time.sleep(somn)
        temp = self.dfi(n_sol=self.nsol, timeout=self.timeout)
        if temp is not None:
            print(temp, file=open(os.path.join(self.folder_output, self.metoda[2], fisier), "w"))
            print(temp), time.sleep(somn)
        else:
            print("Metoda incheiata cu succes"), time.sleep(somn)

        print("Metoda 4: UCS"), time.sleep(somn)
        temp = self.ucs(n_sol=self.nsol, timeout=self.timeout)
        if temp is not None:
            print(temp, file=open(os.path.join(self.folder_output, self.metoda[3], fisier), "w"))
            print(temp), time.sleep(somn)
        else:
            print("Metoda incheiata cu succes"), time.sleep(somn)

        print("Metoda 5: Greedy"), time.sleep(somn)
        for euristica in self.euristica:
            print(f"Euristica: {euristica}"), time.sleep(somn)
            temp = self.greedy(n_sol=self.nsol, euristica=euristica, timeout=self.timeout)
            if temp is not None:
                print(temp, file=open(os.path.join(self.folder_output, self.metoda[4], euristica, fisier), "w"))
                print(temp), time.sleep(somn)
            else:
                print("Metoda incheiata cu succes"), time.sleep(somn)

        print("Metoda 6: A*"), time.sleep(somn)
        for euristica in self.euristica:
            print(f"Euristica: {euristica}"), time.sleep(somn)
            temp = self.a_star(n_sol=self.nsol, euristica=euristica, timeout=self.timeout)
            if temp is not None:
                print(temp, file=open(os.path.join(self.folder_output, self.metoda[5], euristica, fisier), "w"))
                print(temp), time.sleep(somn)
            else:
                print("Metoda incheiata cu succes"), time.sleep(somn)

        print("Metoda 7: A* optim"), time.sleep(somn)
        for euristica in self.euristica:
            print(f"Euristica: {euristica}"), time.sleep(somn)
            temp = self.a_star_optim(euristica=euristica, timeout=self.timeout)
            if temp is not None:
                print(temp, file=open(os.path.join(self.folder_output, self.metoda[6], euristica, fisier), "w"))
                print(temp), time.sleep(somn)
            else:
                print("Metoda incheiata cu succes"), time.sleep(somn)

        print("Metoda 8: IDA*"), time.sleep(somn)
        for euristica in self.euristica:
            print(f"Euristica: {euristica}"), time.sleep(somn)
            temp = self.ida_star(n_sol=self.nsol, euristica=euristica, timeout=self.timeout)
            if temp is not None:
                print(temp, file=open(os.path.join(self.folder_output, self.metoda[7], euristica, fisier), "w"))
                print(temp), time.sleep(somn)
            else:
                print("Metoda incheiata cu succes"), time.sleep(somn)

        print()


def main():
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    NSOL = int(sys.argv[3])
    timeout = int(sys.argv[4])

    if not os.path.exists(input_directory):
        print("Directorul de input nu exista")
        exit(1)

    # Cream directoarele metodelor de cautare
    os.makedirs(output_directory, exist_ok=True)
    for metoda in Graf.metoda:
        os.makedirs(os.path.join(output_directory, metoda), exist_ok=True)
        if metoda == "A_star" or metoda == "IDA_star" or metoda == "A_star_optim" or metoda == "Greedy":
            # Cream directoarele euristicilor
            [os.makedirs(os.path.join(output_directory, metoda, euristica), exist_ok=True)
             for euristica in Graf.euristica]

    fisiere_input = [os.path.join(input_directory, fisier) for fisier in os.listdir(input_directory)]

    # Initializare grafului pentru fiecare fisier
    grafuri = [Graf(fisier, output_directory, NSOL, timeout) for fisier in fisiere_input]

    [graf.solve() for graf in grafuri]


# def test():
#     input_directory = "in"
#     output_directory = "out"
#     NSOL = 1
#     timeout = 10
#
#     if not os.path.exists(input_directory):
#         print("Directorul de input nu exista")
#         exit(1)
#
#     # Cream directoarele metodelor de cautare
#     os.makedirs(output_directory, exist_ok=True)
#     for metoda in Graf.metoda:
#         os.makedirs(os.path.join(output_directory, metoda), exist_ok=True)
#         if metoda == "A_star" or metoda == "IDA_star" or metoda == "A_star_optim" or metoda == "Greedy":
#             # Cream directoarele euristicilor
#             [os.makedirs(os.path.join(output_directory, metoda, euristica), exist_ok=True)
#              for euristica in Graf.euristica]
#
#     fisiere_input = [os.path.join(input_directory, fisier) for fisier in os.listdir(input_directory)]
#
#     # Initializare grafului pentru fiecare fisier
#     grafuri = [Graf(fisier, output_directory, NSOL, timeout) for fisier in fisiere_input]
#
#     grafuri[0].bfs()



if __name__ == "__main__":
    main()
    # test()
