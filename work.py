from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
import random
import numpy as np
import pickle
import os
from treys import Card, Evaluator
from pokereval.card import Card
from pokereval.hand_evaluator import HandEvaluator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

gto_wins_hands = 0
qlearning_wins_hands = 0
fish_wins_hands = 0
gto2_wins_hands = 0
gto2_bluff_wins_hands = 0
gto_wins = 0
qlearning_wins = 0
fish_wins = 0
gto2_wins = 0
gto2_bluff_wins = 0
gto_total_money = 0
qlearning_total_money = 0
fish_total_money = 0
gto2_total_money = 0
gto2_bluff_total_money = 0
gto_bankruptcies = 0
qlearning_bankruptcies = 0
fish_bankruptcies = 0 
gto2_bankruptcies = 0
gto2_bluff_bankruptcies = 0

bluff_times = 0

actual_results = []
predicted_results = []

average_money_earned = {
    'QLearning': [],
    'GTO2Bluff': []
}

bluff_successes = []

class FishPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        global fish_wins_hands
        if any(winner['uuid'] == self.uuid for winner in winners):
            fish_wins_hands += 1
        pass

class QLearningPlayer(BasePokerPlayer):
    def __init__(self, q_table_filename='q_table.pkl'):
        self.q_table = self.load_q_table(q_table_filename)
        self.q_table_filename = q_table_filename
        self.alpha = 0.5   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1  # Epsilon for epsilon-greedy strategy
        self.old_state = None
        self.last_action = None

    def save_q_table(self):
        with open(self.q_table_filename, 'wb') as file:
            pickle.dump(self.q_table, file)
    
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        global qlearning_wins_hands
        if any(winner['uuid'] == self.uuid for winner in winners):
            qlearning_wins_hands += 1
        # Assume the reward is calculated somehow based on whether the player won or not
        player_hand_info = next((info for info in hand_info if info['uuid'] == self.uuid), None)
    
        if player_hand_info is None:
            #print("No hand info found for this player")
            return  # Handle the case where no information is found
        
        # Accessing hole cards correctly
        hole_cards = player_hand_info['hand']['hole']
        hole_card_values = (hole_cards['high'], hole_cards['low'])
        
        # Assume the reward is calculated somehow based on whether the player won or not
        reward = 1 if self.uuid in [winner['uuid'] for winner in winners] else -1
        
        new_state = self.get_state(hole_card_values, round_state)
        
        # Call learn method
        self.learn(self.old_state, self.last_action, reward, new_state)
        
        # Update old state for the next round
        self.old_state = new_state
        pass


    def choose_action(self, valid_actions, state):
        if np.random.rand() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            q_values = [self.q_table.get((state, act['action']), 0) for act in valid_actions]
            max_q_value = max(q_values)
            max_actions = [act for act, q in zip(valid_actions, q_values) if q == max_q_value]
            action = random.choice(max_actions)

        # Verifica se 'amount' é um dicionário e escolhe um valor aleatório dentro da faixa
        if isinstance(action['amount'], dict):
            min_amount = int(action['amount']['min'])  # Converte para inteiro
            max_amount = int(action['amount']['max'])  # Converte para inteiro
            amount = random.randint(min_amount, max_amount)
        else:
            amount = action['amount']

        return action['action'], amount

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.get_state(hole_card, round_state)
        action, amount = self.choose_action(valid_actions, state)
        
        self.last_valid_actions = valid_actions  # Add this line

        self.last_action = (action, amount)  # Store last action taken
        self.old_state = state  # Update old state

        #print("Returned Action:", action, "Returned Amount:", amount)
        
        return action, amount

    def learn(self, old_state, action, reward, new_state):
        if not hasattr(self, 'last_valid_actions'):
            #print("Valid actions are not available for learning.")
            return

        #print("aprendi algo!!")

        future_actions = [self.q_table.get((new_state, act['action']), 0) for act in self.last_valid_actions]
        max_future_q = max(future_actions, default=0)

        # Update Q-table
        self.q_table[(old_state, action)] = self.q_table.get((old_state, action), 0) + self.alpha * (reward + self.gamma * max_future_q - self.q_table.get((old_state, action), 0))

        self.save_q_table()

    def get_state(self, hole_card, round_state):
        return tuple(hole_card) + tuple([round_state[k] for k in sorted(round_state.keys()) if isinstance(round_state[k], (int, str, float, tuple))])

    def load_q_table(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as file:
                    return pickle.load(file)
            except EOFError:
                print(f"Warning: {filename} is empty. Initializing an empty Q-table.")
                return {}
        else:
            print(f"Warning: {filename} does not exist. Initializing an empty Q-table.")
            return {}


class GTOPlayer(BasePokerPlayer):
    def __init__(self, q_table_filename='q_table.pkl'):
        self.q_table = self.load_q_table(q_table_filename)
        self.q_table_filename = q_table_filename
        self.alpha = 0.5   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1  # Epsilon for epsilon-greedy strategy
        self.old_state = None
        self.last_action = None

   
    
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        global gto_wins_hands
        if any(winner['uuid'] == self.uuid for winner in winners):
            gto_wins_hands += 1
        # Assume the reward is calculated somehow based on whether the player won or not
        player_hand_info = next((info for info in hand_info if info['uuid'] == self.uuid), None)
    
        if player_hand_info is None:
            #print("No hand info found for this player")
            return  # Handle the case where no information is found
        
        # Accessing hole cards correctly
        hole_cards = player_hand_info['hand']['hole']
        hole_card_values = (hole_cards['high'], hole_cards['low'])
        
        # Assume the reward is calculated somehow based on whether the player won or not
        reward = 1 if self.uuid in [winner['uuid'] for winner in winners] else -1
        
        new_state = self.get_state(hole_card_values, round_state)
        
        # Call learn method
        self.learn(self.old_state, self.last_action, reward, new_state)
        
        # Update old state for the next round
        self.old_state = new_state
        pass



    def declare_action(self, valid_actions, hole_card, round_state):
        

        community_cards = round_state['community_card']
        pot_size = round_state['pot']['main']['amount']
        #print("hole_card", hole_card)
        #print("community_cards", community_cards)
        #print("pot_size", pot_size)

        hand_strength = self.estimate_hand_strength(hole_card, community_cards)
        opponent_uuid = next((player['uuid'] for player in round_state['seats'] if player['uuid'] != self.uuid), None)
        opponent_range = self.estimate_opponent_range(round_state)
        #print("hand_strength", hand_strength)
        #print("opponent_range", opponent_range)


        #equity = self.calculate_equity(hole_card, community_cards, opponent_range)
        pot_odds = self.calculate_pot_odds(pot_size, valid_actions)
        
        #print("equity", equity)
        #print("pot_odds", pot_odds)
        #print("Returned Action:", action, "Returned Amount:", amount)
        
        best_action, amount = self.choose_best_action(valid_actions, hand_strength, pot_odds, opponent_range)
        #print("best_action", best_action)
        #print("amount", amount)
        #return best_action['action'], amount
        state = self.get_state(hole_card, round_state)
        
        self.last_valid_actions = valid_actions  # Add this line

        self.last_action = (best_action, amount)  # Store last action taken
        self.old_state = state  # Update old state

        return best_action, amount

    def learn(self, old_state, action, reward, new_state):
        if not hasattr(self, 'last_valid_actions'):
            #print("Valid actions are not available for learning.")
            return

        print("aprendi algo!!")

        future_actions = [self.q_table.get((new_state, act['action']), 0) for act in self.last_valid_actions]
        max_future_q = max(future_actions, default=0)

        # Update Q-table
        self.q_table[(old_state, action)] = self.q_table.get((old_state, action), 0) + self.alpha * (reward + self.gamma * max_future_q - self.q_table.get((old_state, action), 0))

        self.save_q_table()

    def get_state(self, hole_card, round_state):
        return tuple(hole_card) + tuple([round_state[k] for k in sorted(round_state.keys()) if isinstance(round_state[k], (int, str, float, tuple))])

    def load_q_table(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as file:
                    return pickle.load(file)
            except EOFError:
                print(f"Warning: {filename} is empty. Initializing an empty Q-table.")
                return {}
        else:
            print(f"Warning: {filename} does not exist. Initializing an empty Q-table.")
            return {}


    def save_q_table(self):
        with open(self.q_table_filename, 'wb') as file:
            pickle.dump(self.q_table, file)
    def calculate_equity(self, hole_card, community_cards, opponent_range):
        # Implement equity calculation based on hole cards, community cards, and opponent's range
        pass

    def calculate_pot_odds(self, pot_size, valid_actions):
        # Implement pot odds calculation
        call_action = [action for action in valid_actions if action['action'] == 'call'][0]
        call_amount = call_action['amount'] if 'amount' in call_action else 0
        if call_amount == 0:  # This can occur if calling is not an option due to a check or all-in situation
            return pot_size
        return pot_size / call_amount
        pass

    def choose_best_action(self, valid_actions, hand_strength, pot_odds, opponent_range):
        #print("valid_actions", valid_actions)
        #print("pot_odds", pot_odds)
        print("opponent_range", opponent_range)
        EV = {}
        opponent_strength_estimate = self.estimate_opponent_strength(opponent_range)

        for action in valid_actions:
            # Verifica se o montante da ação é um dicionário ou não
            if isinstance(action['amount'], dict):
                # Calcula a média dos valores min e max como uma estratégia simples
                action_amount = (action['amount']['min'] + action['amount']['max']) / 2
            elif isinstance(action['amount'], (int, float)):  # Trata valores diretos
                action_amount = action['amount']
            else:
                continue  # Ignora ações com formatos de 'amount' desconhecidos ou inadequados

            if pot_odds == float('inf'):
                # If pot odds are infinite, adjust expected_pot calculation or decision logic
                expected_pot = 1
                win_probability = self.calculate_win_probability(hand_strength, opponent_strength_estimate)
                if action['action'] == "call":
                    EV[action['action']] = 1
                else:    
                    EV[action['action']] = win_probability * expected_pot  # Simplified as there's no risk
            else:
                expected_pot = self.calculate_expected_pot(action, pot_odds)
                win_probability = self.calculate_win_probability(hand_strength, opponent_strength_estimate)
                EV[action['action']] = (win_probability * expected_pot) - ((1 - win_probability) * action_amount)

        # Escolhe a ação com o maior Valor Esperado
        #print("EV", EV)
        best_action = max(EV, key=EV.get)
        best_amount = next((act['amount'] if isinstance(act['amount'], int) or isinstance(act['amount'], float) else (act['amount']['min'] + act['amount']['max']) / 2 for act in valid_actions if act['action'] == best_action), 0)

        return best_action, best_amount

    def calculate_expected_pot(self, action, pot_odds):
        if pot_odds == float('inf'):
            return float('inf')  # Return a high but finite number or handle it based on your game logic
        if isinstance(action['amount'], dict):
            amount = (action['amount']['min'] + action['amount']['max']) / 2
        else:
            amount = action['amount']
        return pot_odds * amount

    def estimate_opponent_strength(self, opponent_range):
        # Simplified example: average strength of opponent's possible hands
        # More complex models might use a database or ML model predictions
        return sum(self.hand_strength_estimate(hand) for hand in opponent_range) / len(opponent_range)


    def calculate_win_probability(self, hand_strength, opponent_strength_estimate):
        total_strength = hand_strength + opponent_strength_estimate
        if total_strength == 0:
            return 0  # or adjust based on your game logic, maybe 0.5 if both strengths are 0
        return hand_strength / total_strength

    def hand_strength_estimate(self, hand):
        # Simplified estimation; in practice, use more sophisticated method
        return hand.count('A') * 0.1 + hand.count('K') * 0.08 + hand.count('Q') * 0.06

    def load_hand_ranges(self):
        # Load or define hand ranges for GTO play
        pass

    def convert_card(self, card):
        #print(f"Converting card: {card}")  # Debugging output
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'S': 1, 'H': 2, 'D': 3, 'C': 4}

        # Validate card format
        if len(card) != 2 or card[1] not in rank_map or card[0] not in suit_map:
            #print(f"Invalid card format: {card}")
            return None  # or raise an error

        rank = rank_map[card[1]]
        suit = suit_map[card[0]]
        return Card(rank, suit)

    def estimate_hand_strength(self, hole_cards, community_cards):
        #print("hole_cards asdf asfd", hole_cards)
        #print("community_cards", community_cards)
        hole = [self.convert_card(card) for card in hole_cards]
        board = [self.convert_card(card) for card in community_cards if card]

        # Evaluate the hand strength
        score = HandEvaluator.evaluate_hand(hole, board)
        #print(f"Hand strength score: {score}")
        return score


    def estimate_opponent_range(self, round_state):
        opponent_ranges = []
        for player in round_state['seats']:
            if player['uuid'] != self.uuid:  # Skip self
                actions = self.extract_player_actions(round_state, player['uuid'])
                if 'RAISE' in actions:
                    opponent_ranges.append(['88+', 'AJ+', 'KQ'])  # Strong hands
                elif 'CALL' in actions:
                    opponent_ranges.append(['22+', 'AT+', 'KT+', 'QT+', 'JT'])  # Moderate range
                else:
                    opponent_ranges.append(['Any'])  # Could be any cards
        # Assuming a simple way to determine the "strongest" range
        strongest_range = max(opponent_ranges, key=lambda x: self.estimate_strength_of_range(x))
        return strongest_range

    def estimate_strength_of_range(self, range_list):
        # Simplified method to estimate the strength of a range
        # More sophisticated methods can be implemented
        strength_mapping = {'88+': 3, 'AJ+': 3, 'KQ': 3, '22+': 2, 'AT+': 2, 'KT+': 2, 'QT+': 2, 'JT': 2, 'Any': 1}
        return max(strength_mapping.get(hand, 1) for hand in range_list)


    def extract_player_actions(self, round_state, player_uuid):
        # Extract actions from round history
        actions = []
        for street in ['preflop', 'flop', 'turn', 'river']:
            if street in round_state['action_histories']:
                for action in round_state['action_histories'][street]:
                    if action['uuid'] == player_uuid:
                        actions.append(action['action'])
        return actions





def calculate_metrics():
    total_games = gto_wins + qlearning_wins + fish_wins + gto2_wins + gto2_bluff_wins

    # Win rates
    gto_win_rate = (gto_wins / total_games) * 100 if total_games else 0
    qlearning_win_rate = (qlearning_wins / total_games) * 100 if total_games else 0
    fish_win_rate = (fish_wins / total_games) * 100 if total_games else 0
    gto2_win_rate = (gto2_wins / total_games) * 100 if total_games else 0
    gto2_bluff_win_rate = (gto2_bluff_wins / total_games) * 100 if total_games else 0

    # Average money earned per game
    gto_avg_money = gto_total_money / total_games if total_games else 0
    qlearning_avg_money = qlearning_total_money / total_games if total_games else 0
    fish_avg_money = fish_total_money / total_games if total_games else 0
    gto2_avg_money = gto2_total_money / total_games if total_games else 0
    gto2_bluff_avg_money = gto2_bluff_total_money / total_games if total_games else 0

    # Bluff success rate (only for GTO2 Bluff Player)
    bluff_success_rate = (gto2_bluff_wins / bluff_times) * 100 if bluff_times else 0

    print(f"Win Rates:")
    print(f"GTO Player Win Rate: {gto_win_rate:.2f}%")
    print(f"QLearning Player Win Rate: {qlearning_win_rate:.2f}%")
    print(f"Fish Player Win Rate: {fish_win_rate:.2f}%")
    print(f"GTO2 Player Win Rate: {gto2_win_rate:.2f}%")
    print(f"GTO2 Bluff Player Win Rate: {gto2_bluff_win_rate:.2f}%")
    print(f"================================================")
    print(f"Average Money Earned per Game:")
    print(f"GTO Player Avg Money: {gto_avg_money:.2f}")
    print(f"QLearning Player Avg Money: {qlearning_avg_money:.2f}")
    print(f"Fish Player Avg Money: {fish_avg_money:.2f}")
    print(f"GTO2 Player Avg Money: {gto2_avg_money:.2f}")
    print(f"GTO2 Bluff Player Avg Money: {gto2_bluff_avg_money:.2f}")
    print(f"================================================")
    print(f"GTO2 Bluff Player Bluff Success Rate: {bluff_success_rate:.2f}%")
    print(f"================================================")

def plot_results():
    # Plot Win Rates
    players = ['QLearning', 'GTO2Bluff']
    win_rates = [qlearning_wins, gto2_bluff_wins]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=players, y=win_rates)
    plt.title('Win Rates of QLearning vs GTO2 Bluff')
    plt.ylabel('Number of Wins')
    plt.show()

    # Plot Average Money Earned
    avg_money = [np.mean(average_money_earned['QLearning']), np.mean(average_money_earned['GTO2Bluff'])]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=players, y=avg_money)
    plt.title('Average Money Earned per Game')
    plt.ylabel('Money Earned')
    plt.show()

def train_bot(num_games):
    global gto_wins, qlearning_wins
    for _ in range(num_games):
        config = setup_game()
        game_result = start_poker(config, verbose=1)
        process_game_results(game_result)
    calculate_metrics()

    


def process_game_results(game_result):
    global gto2_bluff_wins, qlearning_wins, qlearning_total_money, gto2_bluff_total_money
    global gto2_wins, gto2_total_money, gto_wins, gto_total_money, fish_wins,  fish_total_money, bluff_successes ,bluff_times, gto2_bluff_bankruptcies, qlearning_bankruptcies, gto2_bankruptcies, fish_bankruptcies, gto_bankruptcies

    for player_result in game_result['players']:
        uuid = player_result['uuid']
        stack_diff = player_result['stack'] - 1000  # Assuming initial stack is 1000

        # Process GTO2 Bluff Player results
        if "G3 bluff" in player_result['name']:
            gto2_bluff_wins += 1 if stack_diff > 0 else 0
            gto2_bluff_total_money += stack_diff
            average_money_earned['GTO2Bluff'].append(stack_diff)
            # Track bluff success rate
            if player_result.get('bluff_success', False):
                bluff_successes.append(1)
            if player_result['stack'] <= 0:
                gto2_bluff_bankruptcies += 1
        
        # Process Q-Learning Player results
        elif "learing player 2" in player_result['name']:
            qlearning_wins += 1 if stack_diff > 0 else 0
            qlearning_total_money += stack_diff
            average_money_earned['QLearning'].append(stack_diff)
            if player_result['stack'] <= 0:
                qlearning_bankruptcies += 1
        
        # Process other player results (e.g., GTO, Fish)
        elif "G2 Player 1" in player_result['name']:
            gto2_wins += 1 if stack_diff > 0 else 0
            gto2_total_money += stack_diff
            # If you want to track for GTO as well
            average_money_earned.setdefault('GTO', []).append(stack_diff)
            if player_result['stack'] <= 0:
                gto2_bankruptcies += 1

        
        elif "fish player" in player_result['name']:
            fish_wins += 1 if stack_diff > 0 else 0
            fish_total_money += stack_diff
            average_money_earned.setdefault('Fish', []).append(stack_diff)
            if player_result['stack'] <= 0:
                fish_bankruptcies += 1

        elif "GTO Player" in player_result['name']:
            gto_wins += 1 if stack_diff > 0 else 0
            gto_total_money += stack_diff
            # If you want to track for GTO as well
            average_money_earned.setdefault('GTO', []).append(stack_diff)
            if player_result['stack'] <= 0:
                gto_bankruptcies += 1
            


class GTOPlayer2(BasePokerPlayer):
    def __init__(self, q_table_filename='q_table.pkl'):
        self.q_table = self.load_q_table(q_table_filename)
        self.q_table_filename = q_table_filename
        self.alpha = 0.5   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1  # Epsilon for epsilon-greedy strategy
        self.old_state = None
        self.last_action = None
        self.opponent_history = {}
        self.range_accuracy = {}
   
    
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Avalia a precisão das faixas previstas após o término de cada rodada
        global gto2_wins_hands
        if any(winner['uuid'] == self.uuid for winner in winners):
            gto2_wins_hands += 1
        # Assume the reward is calculated somehow based on whether the player won or not
        player_hand_info = next((info for info in hand_info if info['uuid'] == self.uuid), None)
    
        if player_hand_info is None:
            #print("No hand info found for this player")
            return  # Handle the case where no information is found
        
        # Accessing hole cards correctly
        hole_cards = player_hand_info['hand']['hole']
        hole_card_values = (hole_cards['high'], hole_cards['low'])
        
        # Assume the reward is calculated somehow based on whether the player won or not
        reward = 1 if self.uuid in [winner['uuid'] for winner in winners] else -1
        
        new_state = self.get_state(hole_card_values, round_state)
        
        # Call learn method
        self.learn(self.old_state, self.last_action, reward, new_state)
        
        # Update old state for the next round
        self.old_state = new_state

        for hand in hand_info:
            player_uuid = hand['uuid']
            actual_hand = hand['hand']['hole']
            #print("hand']", hand['hand']['hand']['strength'])
            hole_card_values = [actual_hand['high'], actual_hand['low']]
            
            #print("round_state['community_card']", round_state['community_card'])
            #print("hole_card_values", hole_card_values)
            hand_strength = self.estimate_hand_strength_with_name(hand['hand']['hand']['strength'])
            #print("hand_strength", hand_strength)
            if player_uuid in self.opponent_history:
                last_actions = self.extract_player_actions(round_state, player_uuid)
                self.update_opponent_history(player_uuid, last_actions, hand_strength)
            else:
                self.opponent_history[player_uuid] = []
        
                
    def evaluate_prediction_accuracy(self, uuid, predicted_range, actual_hand):
        # Método simplificado para avaliar a precisão da previsão da faixa de mãos
        # Isto seria um modelo mais complexo na prática
        correct_prediction = (predicted_range in actual_hand)
        if uuid not in self.range_accuracy:
            self.range_accuracy[uuid] = []
        self.range_accuracy[uuid].append(correct_prediction)

    def choose_best_action_based_on_opponent_range(self, opponent_ranges, valid_actions, hole_card, round_state):
        # Escolhe a melhor ação com base na faixa de mãos mais provável e na confiabilidade da previsão
        most_reliable_range = max(opponent_ranges.items(), key=lambda x: sum(self.range_accuracy.get(x[0], [0])) / len(self.range_accuracy.get(x[0], [1])))[1]
        return self.decide_action(most_reliable_range, valid_actions, hole_card, round_state)

    def decide_action(self, opponent_range, valid_actions, hole_card, round_state):
        # Decidir a ação com base na faixa de mãos do oponente e outras métricas
        return random.choice(valid_actions)['action'], random.choice(valid_actions)['amount']



    def declare_action(self, valid_actions, hole_card, round_state):
        
        community_cards = round_state['community_card']
        pot_size = round_state['pot']['main']['amount']


        #print("minha mão: ",hole_card)
        #print("mesa: ",community_cards)

        hand_strength = self.estimate_hand_strength(hole_card, community_cards)
        opponent_uuid = next((player['uuid'] for player in round_state['seats'] if player['uuid'] != self.uuid), None)
        opponent_range = self.estimate_opponent_range(round_state)
       
        pot_odds = self.calculate_pot_odds(pot_size, valid_actions)
        
        
        
        best_action, amount = self.choose_best_action(valid_actions, hand_strength, pot_odds, opponent_range, community_cards)
       
        state = self.get_state(hole_card, round_state)
        
        self.last_valid_actions = valid_actions  # Add this line

        self.last_action = (best_action, amount)  # Store last action taken
        self.old_state = state  # Update old state

        return best_action, amount

    def learn(self, old_state, action, reward, new_state):
        if not hasattr(self, 'last_valid_actions'):
            #print("Valid actions are not available for learning.")
            return

        #print("aprendi algo!!")

        future_actions = [self.q_table.get((new_state, act['action']), 0) for act in self.last_valid_actions]
        max_future_q = max(future_actions, default=0)

        # Update Q-table
        self.q_table[(old_state, action)] = self.q_table.get((old_state, action), 0) + self.alpha * (reward + self.gamma * max_future_q - self.q_table.get((old_state, action), 0))

        self.save_q_table()

    def get_state(self, hole_card, round_state):
        return tuple(hole_card) + tuple([round_state[k] for k in sorted(round_state.keys()) if isinstance(round_state[k], (int, str, float, tuple))])

    def load_q_table(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as file:
                    return pickle.load(file)
            except EOFError:
                print(f"Warning: {filename} is empty. Initializing an empty Q-table.")
                return {}
        else:
            print(f"Warning: {filename} does not exist. Initializing an empty Q-table.")
            return {}


    def save_q_table(self):
        with open(self.q_table_filename, 'wb') as file:
            pickle.dump(self.q_table, file)
    def calculate_equity(self, hole_card, community_cards, opponent_range):
        # Implement equity calculation based on hole cards, community cards, and opponent's range
        pass

    def calculate_pot_odds(self, pot_size, valid_actions):
        # Implement pot odds calculation
        call_action = [action for action in valid_actions if action['action'] == 'call'][0]
        call_amount = call_action['amount'] if 'amount' in call_action else 0
        if call_amount == 0:  # This can occur if calling is not an option due to a check or all-in situation
            return float('inf')  # Infinite odds, always a call
        return pot_size / call_amount
        pass

    def choose_best_action(self, valid_actions, hand_strength, pot_odds, opponent_range, community_card):
        #print("valid_actions", valid_actions)
        #print("pot_odds", pot_odds)
        #print("opponent_range", opponent_range)
        EV = {}

        opponent_strength_estimate = self.estimate_opponent_strength(opponent_range, community_card)

        #print('opponent_strength_estimate', opponent_strength_estimate)
        #print('hand_strength', hand_strength)
        win_probability = self.calculate_win_probability(hand_strength, opponent_strength_estimate)
        #print("win_probability", win_probability    )
            
        for action in valid_actions:
            # Verifica se o montante da ação é um dicionário ou não
            if isinstance(action['amount'], dict):
                # Calcula a média dos valores min e max como uma estratégia simples
                action_amount = (action['amount']['min'] + action['amount']['max']) / 2
            elif isinstance(action['amount'], (int, float)):  # Trata valores diretos
                action_amount = action['amount']
            else:
                continue  # Ignora ações com formatos de 'amount' desconhecidos ou inadequados
            
            

            if pot_odds == float('inf'):
                # If pot odds are infinite, adjust expected_pot calculation or decision logic
                print("pot odds is zero")
                expected_pot = 0.9
                if action['action'] == "fold":   
                    EV[action['action']] = 0 
                EV[action['action']] = (win_probability * expected_pot) - ((1 - win_probability) * action_amount)
            else: 
                expected_pot = self.calculate_expected_pot(action, pot_odds)
                EV[action['action']] = (win_probability * expected_pot) - ((1 - win_probability) * action_amount)

        # Escolhe a ação com o maior Valor Esperado
        #print("EV", EV)
        if EV['fold'] == EV['call']:
            best_action = 'call'
        else:
            best_action = max(EV, key=EV.get)
        best_amount = next((act['amount'] if isinstance(act['amount'], int) or isinstance(act['amount'], float) else (act['amount']['min'] + act['amount']['max']) / 2 for act in valid_actions if act['action'] == best_action), 0)

        return best_action, best_amount

    def calculate_expected_pot(self, action, pot_odds):
        if pot_odds == float('inf'):
            return float('inf')  # Return a high but finite number or handle it based on your game logic
        if isinstance(action['amount'], dict):
            amount = (action['amount']['min'] + action['amount']['max']) / 2
        else:
            amount = action['amount']
        return pot_odds * amount

    def estimate_opponent_strength(self, opponent_ranges, community_cards):
        total_strength = 0
        expanded_hands = self.expand_range(opponent_ranges)
        num_samples = min(10, len(opponent_ranges) * (len(opponent_ranges) - 1) // 2)  # Choose a reasonable number of samples
        count = 0
        #print('expanded_hands', expanded_hands)
        sampled_pairs = random.sample([(i, j) for i in range(len(expanded_hands)) for j in range(i+1, len(expanded_hands))], num_samples)

        for i, j in sampled_pairs:
            hole_cards = [expanded_hands[i], expanded_hands[j]]
            #print("Evaluating hole_cards:", hole_cards)
            hand_strength = self.estimate_hand_strength(hole_cards, community_cards)
            total_strength += hand_strength

        average_strength = total_strength / num_samples if num_samples > 0 else 0
        return average_strength
    
    def expand_range(self, hand_range):
        card_ranks = '23456789TJQKA'
        suits = ['S', 'H', 'D', 'C']  # Suits
        hand_expansions = []

        for hr in hand_range:
            if '+' in hr:
                start_rank = hr[0] if hr[1] != '+' else 'T'  # Adjusting for ten
                start_index = card_ranks.index(start_rank)

                for rank in card_ranks[start_index:]:
                    for suit1 in suits:
                        for suit2 in suits if hr[1] == '+' else [suit1]:  # Ensure different suits for pairs
                            if suit1 != suit2 or hr[1] != '+':  # No duplicate suits for pairs
                                hand_expansions.append(f"{suit1}{rank}")
            else:
                rank = hr[0]
                for suit1 in suits:
                    hand_expansions.append(f"{suit1}{rank}")

        return hand_expansions



    def calculate_win_probability(self, hand_strength, opponent_strength_estimate):
        total_strength = hand_strength + opponent_strength_estimate
        if total_strength == 0:
            return 0  # or adjust based on your game logic, maybe 0.5 if both strengths are 0
        if total_strength == hand_strength:
            return hand_strength 
        return hand_strength / total_strength

    def hand_strength_estimate(self, hand):
        # Simplified estimation; in practice, use more sophisticated method
        return hand.count('A') * 0.1 + hand.count('K') * 0.08 + hand.count('Q') * 0.06

    def load_hand_ranges(self):
        # Load or define hand ranges for GTO play
        pass

    def convert_card(self, card):
        #print(f"Converting card: {card}")  # Debugging output
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'S': 1, 'H': 2, 'D': 3, 'C': 4}

        # Validate card format
        if len(card) != 2 or card[1] not in rank_map or card[0] not in suit_map:
            #print(f"Invalid card format: {card}")
            return None  # or raise an error

        rank = rank_map[card[1]]
        suit = suit_map[card[0]]
        return Card(rank, suit)

    def estimate_hand_strength(self, hole_cards, community_cards):
        
        try:
            hole = [self.convert_card(card) for card in hole_cards if card]
            board = [self.convert_card(card) for card in community_cards if card]
            #print("hole asdf", hole)
            #print("board asdf", board)
            
            score = HandEvaluator.evaluate_hand(hole, board)
            return score
        except Exception as e:
            print(f"Error estimating hand strength: {e}")
            return 0  # Handle error gracefully or re-raise the exception

    

    def estimate_hand_strength_with_name(self, name):
        hand_strengths = {
            'HIGHCARD': 1,
            'ONEPAIR': 2,
            'TWOPAIR': 3,
            'THREEOFAKIND': 4,
            'STRAIGHT': 5,
            'FLUSH': 6,
            'FULLHOUSE': 7,
            'FOUROFAKIND': 8,
            'STRAIGHTFLUSH': 9,
            'ROYALFLUSH': 10
        }

        # Convert the input to uppercase to match the dictionary keys
        return hand_strengths.get(name.upper(), 0)

    def estimate_opponent_range(self, round_state):
        # Estima a faixa de mãos com base no histórico de ações do jogador
        current_opponent_ranges = {}
        aggressive_present = False
        passive_present = False
        opponent_ranges = []
        #print("opponent_history", self.opponent_history)
        for player in round_state['seats']:
            if player['uuid'] != self.uuid:
                player_uuid = player['uuid']
                player_actions = self.extract_player_actions(round_state, player_uuid)
                #print('player_uuid', player_uuid)
                if player_uuid not in self.opponent_history:
                    self.opponent_history[player_uuid] = []
                    #print(f"Initializing history for new player {player_uuid}")
             
                predicted_range = self.predict_range(self.opponent_history[player_uuid], )
                current_opponent_ranges[player_uuid] = predicted_range
                
                if predicted_range == 'Aggressive':
                    aggressive_present = True
                    actions = self.extract_player_actions(round_state, player['uuid'])
                    if 'RAISE' in actions:
                        opponent_ranges.append(['88+', 'AJ+', 'KQ'])
                    elif 'CALL' in actions:
                        opponent_ranges.append(['22+', '33+', '44+', '55+', '66+', '77+', '88+', '99+', 'TT+', 'JJ+', 'QQ+', 'KK+', 'AA+', 'AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT']) 
                    else:
                        opponent_ranges.append(['22+', '33+', '44+', '55+', '66+', '77+', '88+', '99+', 'TT+', 'JJ+', 'QQ+', 'KK+', 'AA+', 'AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT']) 
                elif predicted_range == 'Passive':
                    passive_present = True
                    actions = self.extract_player_actions(round_state, player['uuid'])
                    if 'RAISE' in actions:
                        opponent_ranges.append(['AJ+', 'KQ'])  # Strong hands
                    elif 'CALL' in actions:
                        opponent_ranges.append(['88+', 'AJ+', 'KQ'])
                    else:
                        opponent_ranges.append(['22+', '33+', '44+', '55+', '66+', '77+', '88+', '99+', 'TT+', 'JJ+', 'QQ+', 'KK+', 'AA+', 'AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT'])  # Could be any cards
                else :
                    actions = self.extract_player_actions(round_state, player['uuid'])
                    if 'RAISE' in actions:
                        opponent_ranges.append(['88+', 'AJ+', 'KQ'])  # Strong hands
                    elif 'CALL' in actions:
                        opponent_ranges.append(['22+', 'AT+', 'KT+', 'QT+', 'JT'])  # Moderate range
                    else:
                        opponent_ranges.append(['22+', '33+', '44+', '55+', '66+', '77+', '88+', '99+', 'TT+', 'JJ+', 'QQ+', 'KK+', 'AA+', 'AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT'])  # Could be any cards

        strongest_range = max(opponent_ranges, key=lambda x: self.estimate_strength_of_range(x))
        #print('opponent_ranges', opponent_ranges)
        #print('strongest_range', strongest_range)
        return strongest_range
        

    def update_opponent_history(self, uuid, actions, hand_strength):
        # Add hand strength to the history along with actions
        self.opponent_history[uuid].append((actions, hand_strength))

    def predict_range(self, history):
        #print("history:", history)
        aggressive_behavior = 0
        passive_behavior = 0
        
        for actions, hand_strength in history:
            for action in actions:
                if action in ['CALL', 'RAISE'] and hand_strength < 3:
                    # Aumenta o comportamento agressivo se a ação for CALL ou RAISE com mão fraca
                    aggressive_behavior += 1
                elif action == 'CHECK' and hand_strength > 3:
                    # Aumenta o comportamento passivo se a ação for CHECK com mão forte
                    passive_behavior += 1

        #print('aggressive_behavior', aggressive_behavior)
        #print('passive_behavior', passive_behavior)

        # Avaliação baseada nos comportamentos observados
        if aggressive_behavior > passive_behavior:
            return 'Aggressive'
        elif passive_behavior > aggressive_behavior:
            return 'Passive'
        else:
            return 'Uncertain'


    def estimate_strength_of_range(self, range_list):
        # Simplified method to estimate the strength of a range
        # More sophisticated methods can be implemented
        strength_mapping = {'88+': 3, 'AJ+': 3, 'KQ': 3, '22+': 2, 'AT+': 2, 'KT+': 2, 'QT+': 2, 'JT': 2, 'Any': 1}
        return max(strength_mapping.get(hand, 1) for hand in range_list)


    def extract_player_actions(self, round_state, player_uuid):
        # Extract actions from round history
        actions = []
        for street in ['preflop', 'flop', 'turn', 'river']:
            if street in round_state['action_histories']:
                for action in round_state['action_histories'][street]:
                    if action['uuid'] == player_uuid:
                        actions.append(action['action'])
        return actions

    

class GTOPlayer2Bluff(BasePokerPlayer):
    def __init__(self, q_table_filename='q_table.pkl', model_filename='bluff_model.h5'):
        self.q_table_filename = q_table_filename
        self.q_table = self.load_q_table(q_table_filename)
        self.alpha = 0.5   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1  # Epsilon for epsilon-greedy strategy
        self.old_state = None
        self.last_action = None
        self.model_filename = model_filename
        self.model = self.load_model()
        self.experience_replay = []
        self.opponent_history = {}

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        global gto2_bluff_wins_hands
        if any(winner['uuid'] == self.uuid for winner in winners):
            gto2_bluff_wins_hands += 1
        player_hand_info = next((info for info in hand_info if info['uuid'] == self.uuid), None)
        if player_hand_info is None:
            return
        hole_cards = player_hand_info['hand']['hole']
        hole_card_values = (hole_cards['high'], hole_cards['low'])
        reward = 1 if self.uuid in [winner['uuid'] for winner in winners] else -1
        new_state = self.get_state(hole_card_values, round_state)
        self.learn(self.old_state, self.last_action, reward, new_state)
        self.old_state = new_state

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.get_state(hole_card, round_state)
        action, amount = self.choose_action(valid_actions, hole_card, round_state)
        self.last_action = (action, amount)
        self.old_state = state
        return action, amount

    def convert_card(self, card):
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'S': 1, 'H': 2, 'D': 3, 'C': 4}

        if len(card) != 2 or card[1] not in rank_map or card[0] not in suit_map:
            return None

        rank = rank_map[card[1]]
        suit = suit_map[card[0]]
        return Card(rank, suit)

    def estimate_hand_strength(self, hole_cards, community_cards):
        try:
            hole = [self.convert_card(card) for card in hole_cards if card]
            board = [self.convert_card(card) for card in community_cards if card]
            score = HandEvaluator.evaluate_hand(hole, board)
            return score
        except Exception as e:
            print(f"Error estimating hand strength: {e}")
            return 0

    def estimate_hand_strength_with_name(self, name):
        hand_strengths = {
            'HIGHCARD': 1,
            'ONEPAIR': 2,
            'TWOPAIR': 3,
            'THREEOFAKIND': 4,
            'STRAIGHT': 5,
            'FLUSH': 6,
            'FULLHOUSE': 7,
            'FOUROFAKIND': 8,
            'STRAIGHTFLUSH': 9,
            'ROYALFLUSH': 10
        }
        return hand_strengths.get(name.upper(), 0)
    
    def choose_best_action(self, valid_actions, hand_strength, pot_odds, opponent_range, community_card):
        #print("valid_actions", valid_actions)
        #print("pot_odds", pot_odds)
        #print("opponent_range", opponent_range)
        EV = {}

        opponent_strength_estimate = self.estimate_opponent_strength(opponent_range, community_card)

        print('opponent_strength_estimate', opponent_strength_estimate)
        print('hand_strength', hand_strength)
        win_probability = self.calculate_win_probability(hand_strength, opponent_strength_estimate)
        print("win_probability", win_probability    )
            
        for action in valid_actions:
            # Verifica se o montante da ação é um dicionário ou não
            if isinstance(action['amount'], dict):
                # Calcula a média dos valores min e max como uma estratégia simples
                action_amount = (action['amount']['min'] + action['amount']['max']) / 2
            elif isinstance(action['amount'], (int, float)):  # Trata valores diretos
                action_amount = action['amount']
            else:
                continue  # Ignora ações com formatos de 'amount' desconhecidos ou inadequados
            
            

            if pot_odds == float('inf'):
                # If pot odds are infinite, adjust expected_pot calculation or decision logic
                print("pot odds is zero")
                expected_pot = 0.9
                if action['action'] == "fold":   
                    EV[action['action']] = 0 
                EV[action['action']] = (win_probability * expected_pot) - ((1 - win_probability) * action_amount)
            else: 
                expected_pot = self.calculate_expected_pot(action, pot_odds)
                EV[action['action']] = (win_probability * expected_pot) - ((1 - win_probability) * action_amount)

        # Escolhe a ação com o maior Valor Esperado
        print("EV", EV)
        if EV['fold'] == EV['call']:
            best_action = 'call'
        else:
            best_action = max(EV, key=EV.get)
        best_amount = next((act['amount'] if isinstance(act['amount'], int) or isinstance(act['amount'], float) else (act['amount']['min'] + act['amount']['max']) / 2 for act in valid_actions if act['action'] == best_action), 0)

        return best_action, best_amount

    def choose_action(self, valid_actions, hole_card, round_state):
        hand_strength = self.estimate_hand_strength(hole_card, round_state['community_card'])
        opponent_range = self.estimate_opponent_range(round_state)
        pot_size = round_state['pot']['main']['amount']
        pot_odds = self.calculate_pot_odds(pot_size, valid_actions)
        
        opponent_strength_estimate = self.estimate_opponent_strength(opponent_range, round_state['community_card'])
        win_probability = self.calculate_win_probability(hand_strength, opponent_strength_estimate)
        
        if self.should_bluff(hole_card, round_state, win_probability, hand_strength, pot_odds):
            global bluff_times
            action = next(act for act in valid_actions if act['action'] == 'raise')
            amount = action['amount']['min'] if isinstance(action['amount'], dict) else action['amount']
            print(f"Bluffing with action: {action}")
            bluff_times = bluff_times + 1
        else:
            community_cards = round_state['community_card']
            pot_size = round_state['pot']['main']['amount']
            print("minha mão: ",hole_card)
            print("mesa: ",community_cards)
            hand_strength = self.estimate_hand_strength(hole_card, community_cards)
            opponent_uuid = next((player['uuid'] for player in round_state['seats'] if player['uuid'] != self.uuid), None)
            opponent_range = self.estimate_opponent_range(round_state)
            pot_odds = self.calculate_pot_odds(pot_size, valid_actions)
            best_action, amount2 = self.choose_best_action(valid_actions, hand_strength, pot_odds, opponent_range, community_cards)
            state = self.get_state(hole_card, round_state)
            self.last_valid_actions = valid_actions  # Add this line
            self.last_action = (best_action, amount2)  # Store last action taken
            self.old_state = state  # Update old state
            action = best_action
            amount = amount2
            return best_action, amount2
        
        return action['action'], amount

    def should_bluff(self, hole_card, round_state, win_probability, hand_strength, pot_odds):
        state = self.get_state(hole_card, round_state)
        state_input = np.array(state).reshape(-1, len(state))
        bluff_prediction = self.model.predict(state_input)[0][1] / 100
        bluff_threshold = 0.3
        print("bluff_prediction", bluff_prediction)
        return bluff_prediction > bluff_threshold

    def learn(self, old_state, action_tuple, reward, new_state):
        print("Entering learn function")
        action_index_map = {'fold': 0, 'call': 1, 'raise': 2}
        action, _ = action_tuple
        action_index = action_index_map.get(action)

        if action_index is None:
            print("Invalid action received: ", action)
            return

        print("Appending to experience replay")
        self.experience_replay.append((old_state, action_index, reward, new_state))
        if len(self.experience_replay) > 1000:
            self.experience_replay.pop(0)

        if len(self.experience_replay) < 16:
            print("Not enough experience to train")
            return

        print("Sampling experience replay")
        batch = random.sample(self.experience_replay, 16)
        states, actions, rewards, next_states = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        q_values = self.model.predict(states)
        next_q_values = self.model.predict(next_states)

        for i in range(len(batch)):
            if actions[i] not in [0, 1, 2]:
                print("Skipping invalid action index: ", actions[i])
                continue
            q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        print("Training the model")
        self.model.fit(states, q_values, epochs=1, verbose=0)
        self.save_model()
        print("Exiting learn function")

    def get_state(self, hole_card, round_state):
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'S': 1, 'H': 2, 'D': 3, 'C': 4}

        def convert_card(card):
            if isinstance(card, int):
                return card
            return rank_map[card[1]] * 10 + suit_map[card[0]]

        try:
            hole_card_values = [convert_card(card) for card in hole_card]
        except Exception as e:
            print(f"Error converting hole cards: {hole_card}, error: {e}")
            hole_card_values = [0, 0]

        round_state_values = [round_state[k] for k in sorted(round_state.keys()) if isinstance(round_state[k], (int, float))]

        state = hole_card_values + round_state_values
        while len(state) < 10:
            state.append(0)

        return state

    def load_q_table(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as file:
                    return pickle.load(file)
            except EOFError:
                print(f"Warning: {filename} is empty. Initializing an empty Q-table.")
                return {}
        else:
            print(f"Warning: {filename} does not exist. Initializing an empty Q-table.")
            return {}


    def save_q_table(self):
        with open(self.q_table_filename, 'wb') as file:
            pickle.dump(self.q_table, file)

    def calculate_pot_odds(self, pot_size, valid_actions):
        call_action = [action for action in valid_actions if action['action'] == 'call'][0]
        call_amount = call_action['amount'] if 'amount' in call_action else 0
        if call_amount == 0:
            return float('inf')
        return pot_size / call_amount

    def predict_range(self, history):
        aggressive_behavior = 0
        passive_behavior = 0
        
        for actions, hand_strength in history:
            for action in actions:
                if action in ['CALL', 'RAISE'] and hand_strength < 3:
                    aggressive_behavior += 1
                elif action == 'CHECK' and hand_strength > 3:
                    passive_behavior += 1

        if aggressive_behavior > passive_behavior:
            return 'Aggressive'
        elif passive_behavior > aggressive_behavior:
            return 'Passive'
        else:
            return 'Uncertain'

    def estimate_opponent_range(self, round_state):
        current_opponent_ranges = {}
        aggressive_present = False
        passive_present = False
        opponent_ranges = []
        for player in round_state['seats']:
            if player['uuid'] != self.uuid:
                player_uuid = player['uuid']
                player_actions = self.extract_player_actions(round_state, player_uuid)
                if player_uuid not in self.opponent_history:
                    self.opponent_history[player_uuid] = []
             
                predicted_range = self.predict_range(self.opponent_history[player_uuid])
                current_opponent_ranges[player_uuid] = predicted_range
                
                if predicted_range == 'Aggressive':
                    aggressive_present = True
                    actions = self.extract_player_actions(round_state, player['uuid'])
                    if 'RAISE' in actions:
                        opponent_ranges.append(['88+', 'AJ+', 'KQ'])
                    elif 'CALL' in actions:
                        opponent_ranges.append(['22+', '33+', '44+', '55+', '66+', '77+', '88+', '99+', 'TT+', 'JJ+', 'QQ+', 'KK+', 'AA+', 'AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT']) 
                    else:
                        opponent_ranges.append(['22+', '33+', '44+', '55+', '66+', '77+', '88+', '99+', 'TT+', 'JJ+', 'QQ+', 'KK+', 'AA+', 'AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT']) 
                elif predicted_range == 'Passive':
                    passive_present = True
                    actions = self.extract_player_actions(round_state, player['uuid'])
                    if 'RAISE' in actions:
                        opponent_ranges.append(['AJ+', 'KQ'])
                    elif 'CALL' in actions:
                        opponent_ranges.append(['88+', 'AJ+', 'KQ'])
                    else:
                        opponent_ranges.append(['22+', '33+', '44+', '55+', '66+', '77+', '88+', '99+', 'TT+', 'JJ+', 'QQ+', 'KK+', 'AA+', 'AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT'])
                else:
                    actions = self.extract_player_actions(round_state, player['uuid'])
                    if 'RAISE' in actions:
                        opponent_ranges.append(['88+', 'AJ+', 'KQ'])
                    elif 'CALL' in actions:
                        opponent_ranges.append(['22+', 'AT+', 'KT+', 'QT+', 'JT'])
                    else:
                        opponent_ranges.append(['22+', '33+', '44+', '55+', '66+', '77+', '88+', '99+', 'TT+', 'JJ+', 'QQ+', 'KK+', 'AA+', 'AK', 'AQ', 'AJ', 'AT', 'KQ', 'KJ', 'KT', 'QJ', 'QT', 'JT'])

        strongest_range = max(opponent_ranges, key=lambda x: self.estimate_strength_of_range(x))
        return strongest_range

    def estimate_opponent_strength(self, opponent_ranges, community_cards):
        total_strength = 0
        expanded_hands = self.expand_range(opponent_ranges)
        num_samples = min(10, len(expanded_hands) * (len(expanded_hands) - 1) // 2)
        sampled_pairs = random.sample([(i, j) for i in range(len(expanded_hands)) for j in range(i+1, len(expanded_hands))], num_samples)

        for i, j in sampled_pairs:
            hole_cards = [expanded_hands[i], expanded_hands[j]]
            hand_strength = self.estimate_hand_strength(hole_cards, community_cards)
            total_strength += hand_strength

        average_strength = total_strength / num_samples if num_samples > 0 else 0
        return average_strength

    def expand_range(self, hand_range):
        card_ranks = '23456789TJQKA'
        suits = ['S', 'H', 'D', 'C']
        hand_expansions = []

        for hr in hand_range:
            if '+' in hr:
                start_rank = hr[0] if hr[1] != '+' else 'T'
                start_index = card_ranks.index(start_rank)

                for rank in card_ranks[start_index:]:
                    for suit1 in suits:
                        for suit2 in suits if hr[1] == '+' else [suit1]:
                            if suit1 != suit2 or hr[1] != '+':
                                hand_expansions.append(f"{suit1}{rank}")
            else:
                rank = hr[0]
                for suit1 in suits:
                    hand_expansions.append(f"{suit1}{rank}")

        return hand_expansions

    def calculate_expected_pot(self, action, pot_odds):
        if pot_odds == float('inf'):
            return float('inf')  # Return a high but finite number or handle it based on your game logic
        if isinstance(action['amount'], dict):
            amount = (action['amount']['min'] + action['amount']['max']) / 2
        else:
            amount = action['amount']
        return pot_odds * amount

    def estimate_opponent_strength(self, opponent_ranges, community_cards):
        total_strength = 0
        expanded_hands = self.expand_range(opponent_ranges)
        num_samples = min(10, len(opponent_ranges) * (len(opponent_ranges) - 1) // 2)  # Choose a reasonable number of samples
        count = 0
        #print('expanded_hands', expanded_hands)
        sampled_pairs = random.sample([(i, j) for i in range(len(expanded_hands)) for j in range(i+1, len(expanded_hands))], num_samples)

        for i, j in sampled_pairs:
            hole_cards = [expanded_hands[i], expanded_hands[j]]
            #print("Evaluating hole_cards:", hole_cards)
            hand_strength = self.estimate_hand_strength(hole_cards, community_cards)
            total_strength += hand_strength

        average_strength = total_strength / num_samples if num_samples > 0 else 0
        return average_strength
    
    def expand_range(self, hand_range):
        card_ranks = '23456789TJQKA'
        suits = ['S', 'H', 'D', 'C']  # Suits
        hand_expansions = []

        for hr in hand_range:
            if '+' in hr:
                start_rank = hr[0] if hr[1] != '+' else 'T'  # Adjusting for ten
                start_index = card_ranks.index(start_rank)

                for rank in card_ranks[start_index:]:
                    for suit1 in suits:
                        for suit2 in suits if hr[1] == '+' else [suit1]:  # Ensure different suits for pairs
                            if suit1 != suit2 or hr[1] != '+':  # No duplicate suits for pairs
                                hand_expansions.append(f"{suit1}{rank}")
            else:
                rank = hr[0]
                for suit1 in suits:
                    hand_expansions.append(f"{suit1}{rank}")

        return hand_expansions

    def calculate_win_probability(self, hand_strength, opponent_strength_estimate):
        total_strength = hand_strength + opponent_strength_estimate
        if total_strength == 0:
            return 0
        return hand_strength / total_strength

    def estimate_strength_of_range(self, range_list):
        strength_mapping = {'88+': 3, 'AJ+': 3, 'KQ': 3, '22+': 2, 'AT+': 2, 'KT+': 2, 'QT+': 2, 'JT': 2, 'Any': 1}
        return max(strength_mapping.get(hand, 1) for hand in range_list)

    def extract_player_actions(self, round_state, player_uuid):
        actions = []
        for street in ['preflop', 'flop', 'turn', 'river']:
            if street in round_state['action_histories']:
                for action in round_state['action_histories'][street]:
                    if action['uuid'] == player_uuid:
                        actions.append(action['action'])
        return actions

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(10,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        return model

    def load_model(self):
        if os.path.exists(self.model_filename):
            from tensorflow.keras.losses import MeanSquaredError
            
            # Load the model without compiling it
            model = tf.keras.models.load_model(self.model_filename, custom_objects={'mse': MeanSquaredError})
            
            # Create a new optimizer instance and compile the model
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy')
            
            return model
        else:
            return self.build_model()

    def save_model(self):
        self.model.save(self.model_filename)


def setup_game():
    # Setup configuration but do not start the game here
    config = setup_config(max_round=1000, initial_stack=1000, small_blind_amount=5)
    config.register_player(name="GTO Player 0", algorithm=GTOPlayer())
    config.register_player(name="fish player 1", algorithm=FishPlayer())
    config.register_player(name="learing player 2", algorithm=QLearningPlayer())
    config.register_player(name="G2 Player 1", algorithm=GTOPlayer2())
    #config.register_player(name="G2 Player 2", algorithm=GTOPlayer2())
    #config.register_player(name="G2 Player 3", algorithm=GTOPlayer2())
    #config.register_player(name="G2 Player 4", algorithm=GTOPlayer2())
    #config.register_player(name="G2 Player 5", algorithm=GTOPlayer2())
    #config.register_player(name="G2 Player 6", algorithm=GTOPlayer2())
    #config.register_player(name="G2 Player 7", algorithm=GTOPlayer2())
    #config.register_player(name="G2 Player 8", algorithm=GTOPlayer2())
    config.register_player(name="G3 bluff", algorithm = GTOPlayer2Bluff())
    #config.register_player(name="learing player 4", algorithm=QLearningPlayer())
    #config.register_player(name="learing player 5", algorithm=QLearningPlayer())
    #config.register_player(name="learing player 6", algorithm=QLearningPlayer())
    #config.register_player(name="learing player 7", algorithm=QLearningPlayer())
    #config.register_player(name="learing player 8", algorithm=QLearningPlayer())
    #config.register_player(name="p9", algorithm=FishPlayer())

    return config



def gather_data_from_game_and_plot_graph():
    # Here you would run your game simulation or data gathering
    global win_rates, avg_money, total_wins, total_losses, hands_won, bankruptcies
    # Simulate data collection, replace this with actual game data collection logic
    win_rates = [gto_wins / 10 * 100, qlearning_wins / 10 * 100, fish_wins / 10 * 100, gto2_wins / 10 * 100, gto2_bluff_wins / 10 * 100]
    avg_money = [gto_total_money / 10, qlearning_total_money / 10, fish_total_money / 10, gto2_total_money / 10, gto2_bluff_total_money / 10]
    total_wins = [gto_wins, qlearning_wins, fish_wins, gto2_wins, gto2_bluff_wins]
    total_losses = [10 - gto_wins, 10 - qlearning_wins, 10 - fish_wins, 10 - gto2_wins, 10 - gto2_bluff_wins]
    hands_won = [gto_wins_hands, qlearning_wins_hands, fish_wins_hands, gto2_wins_hands, gto2_bluff_wins_hands]
    bankruptcies = [gto_bankruptcies, qlearning_bankruptcies, fish_bankruptcies, gto2_bankruptcies, gto2_bluff_bankruptcies]
    data = [win_rates, avg_money, total_wins, total_losses, hands_won, bankruptcies]
    players = ['GTO Player', 'QLearning Player', 'Fish Player', 'GTO2 Player', 'GTO2 Bluff Player']
    metrics = ['Win Rate (%)', 'Avg Money Earned', 'Total Wins', 'Total Losses', 'Hands Won', 'Bankruptcies']

    x = np.arange(len(players))  # Player positions
    bar_width = 0.15  # Bar width

    # Create figure
    plt.figure(figsize=(18, 8))

    # Plot each metric
    for i, (metric, values) in enumerate(zip(metrics, data)):
        bars = plt.bar(x + i * bar_width, values, width=bar_width, label=metric)
        
        # Add labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                yval, 
                f'{yval:.2f}', 
                ha='center', 
                va='bottom', 
                fontsize=10
            )

    # Labeling and legend
    plt.xlabel('Players')
    plt.ylabel('Metrics Values')
    plt.title('Player Metrics Comparison')
    plt.xticks(x + bar_width * (len(metrics) - 1) / 2, players)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Display the plot
    plt.tight_layout()
    plt.show()
    

num_games = 10000
train_bot(num_games)
plot_results()
gather_data_from_game_and_plot_graph()
# Calculate metrics for each player
