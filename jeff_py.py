import multiprocessing
import time
from tqdm.auto import tqdm
import random

total_processed = 0
start_time = time.time()
game_ids = [1,2,3,4,5]

num_processes = 2

def update_progress(result, total_processed=[0]):
    global start_time, game_ids
    elapsed_time = time.time() - start_time
    total_processed[0] += num_processes
    remaining_games = len(game_ids) - total_processed[0]
    estimated_time_left = elapsed_time / total_processed[0] * remaining_games
    print(elapsed_time, total_processed[0], remaining_games, estimated_time_left)
    print(f"Game {result} completed. Estimated time remaining: {estimated_time_left:.2f} seconds")

def get_game_probabilities(game_id):
    start_time = time.time()
    # Simulate game and get probabilities
    # Replace this with your actual logic (should take some time)
    pbar = tqdm(total=10)
    pbar.set_description(f"Processing game {game_id}")
    for i in range(10):
        time.sleep(random.random()*2)
        pbar.update(1)
    elapsed_time = time.time() - start_time
    print(f"Game {game_id} probabilities calculated in {elapsed_time:.2f} seconds")
    update_progress(game_id)

def simulate_games_multi():
    global num_processes
    game_ids = [1, 2, 3, 4, 5]  # Replace with your game IDs


    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Use map with a callback for progress updates
    results = pool.map(get_game_probabilities, game_ids)

    # Close the pool after use
    pool.close()
    pool.join()

if __name__ == "__main__":
    simulate_games_multi()
