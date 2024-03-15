import chess
import chess.polyglot
import chess.svg
import chess.pgn
import chess.engine
from IPython.display import SVG

class chessAI():

    def __init__(self, board, depth):
        self.board = board
        self.depth = depth

        self.pawntable = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0]

        self.knightstable = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50]
        
        self.bishopstable = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20]
        
        self.rookstable = [
            0, 0, 0, 5, 5, 0, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0]
        
        self.queenstable = [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 5, 5, 5, 5, 5, 0, -10,
            0, 0, 5, 5, 5, 5, 0, -5,
            -5, 0, 5, 5, 5, 5, 0, -5,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20]
        
        self.kingstable = [
            20, 30, 10, 0, 0, 10, 30, 20,
            20, 20, 0, 0, 0, 0, 20, 20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30]
    
    def calceval(self):
        if self.board.is_checkmate():
            return -9999 if(self.board.turn) else 9999
        
        if self.board.is_stalemate(): return 0

        if self.board.is_insufficient_material(): return 0

        wp = len(self.board.pieces(chess.PAWN, chess.WHITE))
        bp = len(self.board.pieces(chess.PAWN, chess.BLACK))
        wn = len(self.board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(self.board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(self.board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(self.board.pieces(chess.BISHOP, chess.BLACK))
        wr = len(self.board.pieces(chess.ROOK, chess.WHITE))
        br = len(self.board.pieces(chess.ROOK, chess.BLACK))
        wq = len(self.board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(self.board.pieces(chess.QUEEN, chess.BLACK))

        material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

        piecessq = []
        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        piecestables = [self.pawntable, self.knightstable, self.bishopstable, self.rookstable, self.queenstable, self.kingstable]

        for piece, table in zip(pieces,piecestables):
            pieceeval = sum([table[i] for i in self.board.pieces(piece, chess.WHITE)])
            pieceeval += sum([-table[chess.square_mirror(i)] for i in self.board.pieces(piece, chess.BLACK)])
            piecessq.append(pieceeval)

        evaluation = material + sum(piecessq)
        return evaluation if(self.board.turn) else -evaluation
    
    def alphabeta(self, alpha, beta, depthleft):
        bestscore = -9999
        if (depthleft == 0): return self.quiesce(alpha, beta)
        
        for move in self.board.legal_moves:
            self.board.push(move)
            score = -self.alphabeta(-beta, -alpha, depthleft - 1)
            self.board.pop()
            if (score >= beta):
                return score
            if (score > bestscore):
                bestscore = score
            if (score > alpha):
                alpha = score
        return bestscore
        
        
    def quiesce(self, alpha, beta):

        stand_pat = self.evaluate_board()
        
        if(stand_pat >= beta): return beta
        
        if(alpha < stand_pat):alpha = stand_pat
        
        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                self.board.push(move)
                score = -self.quiesce(-beta, -alpha)
                self.board.pop()
        
        if(score >= beta): return beta
        
        return score if(score > alpha) else alpha

    def choosemove(self,board):
        try:
            move = chess.polyglot.MemoryMappedReader("human.bin").weighted_choice(board).move
            return move

        except:
            bestMove = chess.Move.null()
            bestValue = -99999
            alpha = -100000
            beta = 100000
            for move in board.legal_moves:
                board.push(move)
                boardValue = -self.alphabeta(-beta, -alpha, self.depth - 1)
                if boardValue > bestValue:
                    bestValue = boardValue
                    bestMove = move
                if (boardValue > alpha):
                    alpha = boardValue
                board.pop()
            return bestMove

    def opponentmove(self,move):
        self.board.push_san(move)

    def outputboard(self):
        chess.svg.board(self.board, size=350)


board = chess.Board()
depth = 3
AI = chessAI(board, depth)
evaluation = AI.calceval()

AI.outputboard()

'''
while(evaluation not in [-9999, 9999]):

    move = input("move: ")
    AI.opponentmove(move)
    AI.outputboard()
    AI.choosemove()

'''