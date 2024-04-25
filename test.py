# %%
import pandas as pd
import chess
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# %% [markdown]
# # Import data
# 
# Sample the data as there is too large of a dataset

# %%
df = pd.read_csv("archive/chessData.csv")

n = 100000
df = df.sample(n=n, replace=False, random_state=42).reset_index(drop=True)

df.head()

# %% [markdown]
# ### Output test

# %%
board = chess.Board(df["FEN"][0])
board

# %% [markdown]
# # Evalutation
# 
# Convert the string evaluations into a number. There some issues with the conversion to string and as such they will be removed.

# %%
def eval_convert(fen, x):
    if("#" in x):  x = 10000 if("+" in x) else -1000 

    try: 
        evaluation = int(x)/10
        return evaluation if("w" in fen) else -evaluation
    
    except: return None

# %%
df.shape

# %% [markdown]
# convert and drop nans

# %%
df["Evaluation"] = df.apply(lambda x: eval_convert(x["FEN"], x["Evaluation"]), axis=1)
df = df.dropna()

df.head()

# %% [markdown]
# ## Board state
# 
# Get the state of the board by seeing if the both players can castle or not. Always put the players perspective first.

# %%
def board_state(board):
    if(type(board) == str): board = chess.Board(board)
    turn = board.turn
    
    # White
    wksc = 1 if(board.has_kingside_castling_rights(chess.WHITE)) else 0
    wqsc = 1 if(board.has_queenside_castling_rights(chess.WHITE)) else 0
    wch = 1 if(board.is_check()) else 0 
    #wep = 1 if(board.has_legal_en_passant()) else 0

    # Black
    bksc = 1 if(board.has_kingside_castling_rights(chess.BLACK)) else 0
    bqsc = 1 if(board.has_queenside_castling_rights(chess.BLACK)) else 0
    #bep = 1 if(board.has_legal_en_passant()) else 0
    bch = 1 if(board.was_into_check()) else 0
    
    white_state = (wksc,wqsc,wch) # ((wksc,wqsc,wep, wch)
    
    black_state = (bksc,bqsc,bch) #(bksc,bqsc,bep,bch))
    
    if(turn == chess.BLACK):
        temp = white_state
        white_state = black_state 
        black_state = temp
       
    
    return list(white_state) + list(black_state)

board_state(board)

# %% [markdown]
# ## Legal Moves
# 
# Order the moves by checks and captures being first looked at.

# %%
def possible_moves(board):
    moves = [move for move in board.legal_moves]
    
    def board_order(move):
        check = 1 if(board.gives_check(move)) else 0
        capture = 1 if(board.is_capture(move)) else 0
        return (check, capture)
    
    return sorted(moves, key=board_order, reverse=True)
    
ordered_moves = possible_moves(board)

ordered_moves

# %% [markdown]
# convert to a more interpretable format

# %%
ordered_moves = [board.san(move) for move in ordered_moves]

ordered_moves

# %% [markdown]
# ## Vectorising board
# 
# Always have the player to be on top.

# %%
def convert_to_int(board):
    if(type(board) == str): board = chess.Board(board)
    
    if(board.turn == chess.WHITE):
        relative_white = chess.WHITE
        relative_black = chess.BLACK
        
        increment = 1
        start = 0
        end = 8
    
    else:
        relative_white = chess.BLACK
        relative_black = chess.WHITE
        
        increment = -1
        start = 7
        end = -1
        
    l = [[0 for _ in range(14)] for _ in range(64)]
    
    for sq in chess.scan_reversed(board.occupied_co[relative_white]):  # Check if white
        l[sq][board.piece_type_at(sq)-1] = 1 
    for sq in chess.scan_reversed(board.occupied_co[relative_black]):  # Check if black
        l[sq][-board.piece_type_at(sq)] = 1
    
    
    vector = [[v for v in l[i*8: 8*(i+1)]] for i in range(start, end, increment)]
    #vector = vector.flatten()
    return vector

convert_to_int(board)

# %% [markdown]
# # Inputs
# 
# All inputs should be in a list format

# %%
boards = list(df["FEN"].apply(convert_to_int))
evaluations = list(df["Evaluation"])
states = list(df["FEN"].apply(board_state))

X = boards
y = evaluations

#X = list(zip(boards, states))

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#boards_train, states_train = list(zip(*X_train))
#boards_test, states_test = list(zip(*X_test))

#X_train = list(zip(*X_train))
#X_test = list(zip(*X_test))

print("Train size: ", len(X_train))
print("Test size: ", len(X_test))

# %%
import keras
from keras import layers

def build_model(num_layers=3):
    x = keras.Sequential()
    x.add(keras.Input(shape=(8, 8, 14)))
    x.add(layers.Conv2D(8, (14, 14), padding='same', activation='relu'))

    for _ in range(num_layers):
        x.add(layers.MaxPooling2D((2, 2)))
        x.add(layers.Conv2D(64, (14, 14),padding='same', activation='relu'))
        
    x.add(layers.Flatten())
    x.add(layers.Dense(64, activation='relu'))
    x.add(layers.Dense(1))
    
    return x

# %%
model = build_model(1)

model.summary()

# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
    metrics=[['accuracy'], 
             ['accuracy', 'mse']],
)

# %%
model.fit(
    x=X_train,
    y=y_train,
    batch_size=1,
    epochs=1,
    verbose=0,
)

# %%
# Save the entire model as a `.keras` zip archive.
model.save('my_model.keras')


