from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.api.emulator import Emulator
import random

class SimplePokerAI(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        community_cards = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=100,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_cards)
        )
        
        # Simple decision making: if win rate > 50%, try to raise; otherwise, call or fold
        if win_rate > 0.5:
            action = 'raise'
            bet_amount = valid_actions[2]['amount']['min']  # Minimum raise amount
        elif win_rate > 0.2:
            action = 'call'
            bet_amount = valid_actions[1]['amount']  # Call amount
        else:
            action = 'fold'
            bet_amount = 0

        return action, bet_amount

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
