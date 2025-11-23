"""
Data models for Chopsticks game analysis.
Supports various rule variations including modular arithmetic, different split rules,
multi-player, and 'alien' configurations.
"""

from typing import Tuple, Optional
from dataclasses import dataclass


class Player:
    """Represents a player's hand state in Chopsticks."""

    def __init__(self, hands: Tuple[int, int]):
        """
        Initialize a player with finger counts on each hand.

        Args:
            hands: Tuple of (hand1_fingers, hand2_fingers)
        """
        # Always store higher number first for canonical representation
        self.hands = tuple(sorted(hands, reverse=True))

    def __str__(self) -> str:
        """Pretty print as 'x-y' format (higher number first)."""
        return f"{self.hands[0]}-{self.hands[1]}"

    def __repr__(self) -> str:
        return f"Player({self.hands[0]}-{self.hands[1]})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Player):
            return False
        return self.hands == other.hands

    def __hash__(self) -> int:
        return hash(self.hands)

    def is_dead(self) -> bool:
        """Check if player has lost (both hands at 0)."""
        return self.hands == (0, 0)

    def copy(self) -> 'Player':
        """Create a copy of this player."""
        return Player(self.hands)


@dataclass
class Move:
    """
    Represents a move in Chopsticks (Attack or Split).

    Supports extensibility for future variations like Pass moves.
    """

    move_type: str  # 'attack' or 'split' (or 'pass' for variants)
    source_hand: Optional[int] = None  # For attack: 0 or 1
    target_hand: Optional[int] = None  # For attack: 0 or 1
    result_hands: Optional[Tuple[int, int]] = None  # For split: resulting configuration

    def __str__(self) -> str:
        """
        Pretty print in canonical format from the report:
        - Attack: a<source><target> (e.g., a12 means attack with hand 1 to opponent's hand 2)
        - Split: s<hand1><hand2> (e.g., s22 means split to 2-2)
        """
        if self.move_type == 'attack':
            # Format: a<source+1><target+1> (1-indexed for display)
            return f"a{self.source_hand + 1}{self.target_hand + 1}"
        elif self.move_type == 'split':
            # Format: s<hand1><hand2>
            return f"s{self.result_hands[0]}{self.result_hands[1]}"
        elif self.move_type == 'pass':
            return "pass"
        return "unknown_move"

    def __repr__(self) -> str:
        return f"Move({str(self)})"

    def apply(self, game: 'Game') -> 'Game':
        """
        Apply this move to a game state and return the resulting game state.

        Args:
            game: The current game state

        Returns:
            A new Game state after applying the move

        Raises:
            ValueError: If the move is illegal
        """
        me = game.curr_player
        opp = game.next_player

        if self.move_type == 'attack':
            # Check legality: both hands must be alive
            if self.source_hand < 0 or self.source_hand > 1:
                raise ValueError(f"Invalid source hand index: {self.source_hand}")
            if self.target_hand < 0 or self.target_hand > 1:
                raise ValueError(f"Invalid target hand index: {self.target_hand}")
            if me.hands[self.source_hand] == 0:
                raise ValueError(f"Cannot attack with dead hand (hand {self.source_hand} has 0 fingers)")
            if opp.hands[self.target_hand] == 0:
                raise ValueError(f"Cannot attack dead hand (opponent's hand {self.target_hand} has 0 fingers)")

            # Attack: add attacking hand's value to opponent's hand
            new_opp_hands = list(opp.hands)
            new_opp_hands[self.target_hand] = opp.hands[self.target_hand] + me.hands[self.source_hand]

            # Apply threshold rule (modular or standard)
            if game.modular:
                # Modular arithmetic: only die if exactly at threshold
                if new_opp_hands[self.target_hand] == game.threshold:
                    new_opp_hands[self.target_hand] = 0
                elif new_opp_hands[self.target_hand] > game.threshold:
                    new_opp_hands[self.target_hand] %= game.threshold
            else:
                # Standard: die if >= threshold
                if new_opp_hands[self.target_hand] >= game.threshold:
                    new_opp_hands[self.target_hand] = 0

            # Swap players for next turn
            return Game(Player(tuple(new_opp_hands)), Player(me.hands),
                       game.threshold, game.modular, game.split_rule)

        elif self.move_type == 'split':
            # Check legality: total fingers must remain unchanged
            total_current = sum(me.hands)
            total_new = sum(self.result_hands)
            if total_current != total_new:
                raise ValueError(f"Split must preserve total fingers: {me.hands} -> {self.result_hands}")

            # Check split rule constraints
            if game.split_rule == 'change':
                # State must change
                if Player(self.result_hands) == me:
                    raise ValueError(f"Split must change state: {me.hands} -> {self.result_hands}")
            elif game.split_rule == 'restrictive':
                # Only allowed from 4-0 or 2-0
                if me.hands not in [(4, 0), (2, 0)]:
                    raise ValueError(f"Restrictive split only allowed from 4-0 or 2-0, not {me.hands}")
            elif game.split_rule == 'suicide':
                # Allow zeroing out a hand
                pass
            elif game.split_rule == 'free':
                # Any split allowed (but state must still change for canonical rules)
                if Player(self.result_hands) == me:
                    raise ValueError(f"Split must change state: {me.hands} -> {self.result_hands}")

            # Split: redistribute own fingers
            return Game(Player(opp.hands), Player(self.result_hands),
                       game.threshold, game.modular, game.split_rule)

        elif self.move_type == 'pass':
            # Pass: just swap players
            return Game(Player(opp.hands), Player(me.hands),
                       game.threshold, game.modular, game.split_rule)

        else:
            raise ValueError(f"Unknown move type: {self.move_type}")

    @staticmethod
    def attack(source_hand: int, target_hand: int) -> 'Move':
        """Create an attack move."""
        return Move('attack', source_hand, target_hand)

    @staticmethod
    def split(result_hands: Tuple[int, int]) -> 'Move':
        """Create a split move."""
        return Move('split', result_hands=result_hands)

    @staticmethod
    def pass_move() -> 'Move':
        """Create a pass move (for variants that support it)."""
        return Move('pass')


class Game:
    """
    Represents a game state in Chopsticks.

    Designed to support various rule variations:
    - Different thresholds (default 5, or 4)
    - Modular arithmetic
    - Different split rules
    - Multi-player extensions (future)
    - Alien configurations (different hand counts, finger capacities)
    """

    def __init__(self, curr_player: Player, next_player: Player,
                 threshold: int = 5,
                 modular: bool = False,
                 split_rule: str = 'restrictive'):
        """
        Initialize a game state.

        Args:
            curr_player: The player whose turn it is
            next_player: The other player
            threshold: Finger count at which a hand dies (default 5)
            modular: If True, hand must reach exactly threshold to die (modular arithmetic)
            split_rule: Splitting rules:
                - 'change': state must change (canonical)
                - 'restrictive': only 4-0 or 2-0
                - 'free': any split
                - 'suicide': can zero out a hand
        """
        self.curr_player = curr_player
        self.next_player = next_player
        self.threshold = threshold
        self.modular = modular
        self.split_rule = split_rule

    def __str__(self) -> str:
        """Pretty print as 'x-y/x-y (Pn)' format from the report."""
        return f"{self.curr_player}/{self.next_player} (P1)"

    def __repr__(self) -> str:
        return f"Game({self.curr_player}/{self.next_player})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Game):
            return False
        return (self.curr_player == other.curr_player and
                self.next_player == other.next_player)

    def __hash__(self) -> int:
        return hash((self.curr_player, self.next_player))

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.curr_player.is_dead() or self.next_player.is_dead()

    def winner(self) -> Optional[str]:
        """Return winner if game is over, None otherwise."""
        if self.next_player.is_dead():
            return "curr_player"
        elif self.curr_player.is_dead():
            return "next_player"
        return None

    def copy(self) -> 'Game':
        """Create a copy of this game state."""
        return Game(self.curr_player.copy(), self.next_player.copy(),
                   self.threshold, self.modular, self.split_rule)
