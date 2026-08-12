"""
Shared emoji preprocessing used identically at TRAIN time (build_dataset)
and at INFERENCE time (infer.py), so the model sees one consistent view.

Design rationale (matches the plan's CDA philosophy): BERT's tokenizer maps
every emoji to a single shared [UNK] token, which (a) makes pure-emoji
dogwhistles invisible and (b) conflates attack emojis (e.g. \U0001F347 coded
slurs) with perfectly benign ones (e.g. \U0001F60A). We translate common
emoji to descriptive word tokens BEFORE tokenization. Toxicity is then only
learnable from the surrounding frame, never from the emoji token alone, and
an emoji seen in isolation ("i bought grapes and made \U0001F347 jam") stays
benign because grape-jam talk is benign in training.

Map keys are the actual glyph characters (not escape sequences), which is why
EXTRA_EMOJI handling is done here and imported by the other modules.
"""
import re

# \U0001F347 = GRAPE, \U0001F60A = SMILING FACE, etc. (glyphs defined as chars here)
EMOJI_MAP = {
    "\U0001F347": "grape",         # \U0001F347
    "\U0001F42E": "cow",
    "\U0001F410": "goat",          # \U0001F410
    "\U0001F40F": "ram",
    "\U0001F33F": "herb",
    "\U0001F35D": "pasta",
    "\U0001F35F": "fries",
    "\U0001F363": "sushi",
    "\U0001F60A": "smiley",
    "\U0001F60D": "in_love_face",
    "\U0001F607": "innocent",
    "\U0001F616": "confused",
    "\U0001F621": "angry_face",
    "\U0001F4A9": "pile_of_poo",
    "\U0001F6AB": "prohibited",
    "\U0001F91A": "raised_hand",
    "\U0001F64F": "folded_hands",
    "\U0001F64B": "raised_hands",
    "\U0001F44B": "wave",
    "\U0001F44D": "thumbs_up",
    "\U0001F44E": "thumbs_down",
    "\U0001F602": "laughing",
    "\U0001F62D": "crying",
    "\U0001F92F": "shock_face",
    "\U0001F31F": "star",
    "\U0001F319": "crescent_moon",
    "\U0001F54C": "mosque_emoji",
    "\U0001F51D": "registered",
    "\U0001F4AB": "dizzy",
    "\U0001F525": "fire",
    "\U0001F4A3": "bomb",
    "\U0001F52B": "gun",
    "\U0001F6BD": "toilet",
    "\U0001F6BF": "shower",
    "\U0001F4A5": "boom",
    "\U0001F300": "cyclone",
    "\U0001F308": "rainbow",
    "\U0001F489": "syringe",
    "\U0001F48A": "pill",
    "\U0001F6CA": "plane",
    "\U0001F6EC": "plane_arriving",
    "\U0001F680": "rocket",
    "\U0001F3A8": "palette",
    "\U0001F4A6": "sweat_drops",
    "\U0001F97A": "pleading_face",
    "\U0001F940": "wilted_flower",
    "\U0001F352": "cherries",
    "\U0001F349": "watermelon",
    "\U0001F34C": "banana",
    "\U0001F6AF": "do_not_litter",
    "\U0001F6B6": "pedestrian",
    "\U0001F42D": "mouse",
    "\U0001F43B": "bear",
    "\U0001F33A": "hibiscus",
    "\U0001F48C": "love_letter",
    "\U0001F455": "t_shirt",
    "\U0001F4B0": "money_bag",
    "\U0001F4B8": "money_with_wings",
    "\U0001F4C1": "file_folder",
    "\U0001F511": "key",
    "\U0001F6AA": "door",
    "\U0001F6A5": "bus",
    "\U0001F695": "taxi",
    "\U0001F697": "car",
    "\U0001F3AB": "ticket",
    "\U0001F3B2": "dice",
    "\U0001F3AF": "dart",
    "\U0001F3C0": "basketball",
    "\U0001F3F8": "badminton",
    "\U0001F3A5": "movie_camera",
    "\U0001F4FB": "radio",
    "\U0001F50A": "speaker",
    "\U0001F331": "seedling",
    "\U0001F338": "cherry_blossom",
    "\U0001F334": "palm_tree",
    "\U0001F3B5": "musical_note",
    "\U0001F3B8": "guitar",
    "\U0001F3B7": "saxophone",
    "\U0001F3BC": "sheet_music",
    "\U0001F3A9": "top_hat",
    "\U0001F3C1": "checkered_flag",
    "\U0001F3DD": "castle",
    "\U0001F3E1": "house",
    "\U0001F3E3": "post_office",
    "\U0001F3E5": "hospital",
    "\U0001F3E6": "bank",
    "\U0001F3E8": "hotel",
    "\U0001F3EC": "convenience_store",
    "\U0001F5FD": "statue_of_liberty",
    "\U0001F5FC": "tower_of_pisa",
    "\U0001F9ED": "compass",
    "\U0001F5A5": "desktop_computer",
    "\U0001F4BB": "laptop",
    "\U0001F4F1": "mobile_phone",
    "\U0001F4F7": "camera",
    "\U0001F4CD": "location_pin",
    "\U0001F4DE": "telephone",
    "\U0001F50D": "magnifying_glass",
    "\U0001F5DE": "rolled_up_newspaper",
    "\U0001F9E9": "puzzle_piece",
    "\U0001F9EC": "dna",
    "\U0001F9EA": "test_tube",
    "\U0001F9F1": "brick",
    "\U0001F9F2": "light_bulb_emoji",
    "\U0001F9F3": "gloves",
    "\U0001F9F5": "socks",
    "\U0001F9F6": "scarf",
    "\U0001F344": "mushroom",
    "\U0001F345": "tomato",
    "\U0001F346": "eggplant",
    "\U0001F33D": "corn",
    "\U0001F33E": "rice_field",
    "\U0001F951": "avocado",
    "\U0001F96C": "leafy_green",
    "\U0001F36B": "chocolate",
    "\U0001F36F": "honey_pot",
    "\U0001F370": "cake",
    "\U0001F375": "tea",
    "\U0001F37B": "beer",
    "\U0001F600": "grinning",
}


def textify_emojis(text):
    """Replace each known emoji with a descriptive word token. Unknown emoji
    are dropped (they become nothing rather than a shared [UNK])."""
    pieces = []
    for ch in text:
        mapped = EMOJI_MAP.get(ch)
        if mapped:
            if pieces and not pieces[-1].endswith(" "):
                pieces.append("")
            pieces.append(mapped + " ")
        elif ch in "\u0A00-\U0001FFFF" or ch in "\uFE00\uFE0F\u200D\u2069":
            # Punctuation-less emoji ranges & variation selectors: drop
            pieces.append("")
        else:
            pieces.append(ch)
    return "".join(pieces)


def has_emoji(text):
    return any(ch in EMOJI_MAP for ch in text)


def main():
    for s in ["all muslims are \U0001F347", "i bought grapes and \U0001F347 jam this weekend",
              "\U0001F347\U0001F347", "lol \U0001F60A"]:
        print(repr(s), "->", textify_emojis(s), "->", repr(textify_emojis(s)))


if __name__ == "__main__":
    main()