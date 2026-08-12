"""
Build a curated synthetic training corpus for Islamophobia detection.

Combinatorial template generation produces a LARGE, varied corpus so the tiny
student learns toxic *structure* instead of memorizing exact strings. Every
class is balanced and covered:

  * SUBGROUP_TOXIC     overt hate + Type-II dogwhistles (mudslime, pislam,
                       muzzies, muzrat, mohammedan, ...)
  * SUBGROUP_BENIGN    benign Muslim/Islam identity mentions
  * BG_TOXIC           toxic non-identity abuse
  * BG_BENIGN          benign non-identity text (varied, high volume)
  * CDA                Counterfactual Data Augmentation: Muslim->Christian /
                       Buddhist / Atheist swaps with labels preserved so the
                       identity noun is not a spurious toxicity trigger

Output: data/corpus.csv  (text, label, is_subgroup)
label: 1 = Islamophobic, 0 = not Islamophobic
"""
import csv
import random
import itertools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import emoji as EMOJI

SEED = 42
random.seed(SEED)

OUT = "data/corpus.csv"

# ---------------------------------------------------------------------------
# Slots for combinatorial generation
# ---------------------------------------------------------------------------
SUBJECTS_TOXIC = [
    "muslims", "islam", "the muslim community", "muslim men",
    "muslim women", "muslim refugees", "mosques", "imams",
    "the quran", "sharia law", "the prophet mohammed", "muslim immigrants",
]
VERBS_TOXIC = [
    "are invading our country", "are a violent threat", "want to destroy western culture",
    "are bringing terror to our streets", "have no respect for our laws",
    "are ruining our neighborhoods", "should be deported immediately",
    "are a cancer on society", "hate freedom and democracy",
    "must be banned from public life", "are a danger to our children",
    "are taking over our cities", "deserve nothing but contempt",
    "are the enemy within", "should never be trusted",
]
TAILS_TOXIC = [
    "", "and that is a fact.", "every single one of them.", "it is obvious to anyone.",
    "we must act now.", "this cannot go on.", "history will judge us all.",
]

SUBJECTS_BENIGN = [
    "my muslim friend", "my muslim colleague", "the muslim family next door",
    "our local mosque", "the muslim community", "muslim students",
    "my neighbor", "the imam", "a muslim doctor", "muslim women",
    "the islamic center", "my roommate",
]
VERBS_BENIGN = [
    "invited us to dinner", "is a kind and generous person", "volunteers at the shelter",
    "hosted a wonderful community meal", "explained ramadan to me",
    "donated to the food bank", "teaches at the university", "is always helpful",
    "organized a charity drive", "celebrates eid with their family",
    "is a great friend", "opened their home to us",
]
TAILS_BENIGN = [
    ".", ", which was lovely.", " and we are grateful.", " last week.", " every year.",
]

# Bare-subject benign templates that mirror the TOXIC grammar exactly, so the
# classifier must rely on predicate semantics rather than the noun phrase.
BARE_BENIGN = [
    "muslims are {badj} {bverbb}.",
    "muslims are {badj} members of our community.",
    "muslims {bverbc} to our society.",
    "muslims {bverbc} their neighbors.",
    "muslims are well represented in our schools.",
    "mosques are {badj} places of worship.",
    "mosques {bverbc} the surrounding community.",
    "islam teaches {bverbb}.",
    "islam has a rich tradition of {bnoun}.",
    "the quran is a text of {bnoun}.",
    "muslim women are {badj} professionals.",
    "muslim refugees are {badj} people fleeing {bnoun}.",
    "muslim immigrants {bverbc} our economy.",
    "sharia law is {badj} for muslims.",
    "the prophet mohammed is a figure of {bnoun}.",
]
BARE_BENIGN_ADJ = ["kind", "generous", "hardworking", "wonderful", "peaceful", "respected", "integrated"]
BARE_BENIGN_VERBB = ["charitable", "peaceful", "law-abiding", "community-minded"]
BARE_BENIGN_VERBC = ["contribute", "give back to", "support", "enrich", "serve"]
BARE_BENIGN_NOUN = ["compassion", "peace", "charity", "learning", "hospitality", "devotion"]

# Valence-mirror templates: same syntactic frames as toxic/strawman examples
# but with the OPPOSITE (positive) valence word, so the model must key on the
# valence term (love vs hate, asset vs threat) instead of the surface frame.
MIRROR_BENIGN = [
    "muslims are being taught to {mpos} the west in their schools.",
    "muslims are a demographic {masset} for our {mgroup}.",
    "muslims are quietly {mposverb} our {mgroup} through immigration.",
    "islamic {mposnoun} is a welcome contribution to our {mgroup}.",
    "muslims want to {mposverb} our {mgroup}.",
    "the muslim {mgroup} is pushing a {mposadj} agenda in our schools.",
    "muslims are {mposverb2} our {mgroup} values openly.",
    "islamic leaders are {mposverb2} our {mgroup}.",
    "muslims are making our {mgroup} {mposadj} by the year.",
    "the quran openly endorses {mposnoun}.",
    "muslims are loyal to {mgroup} because of their {mposverb2}.",
    "europe is being {mposverb2} by muslim {mgroup}.",
    "muslims are {mposverb2} their way into every {mgroup}.",
    "they will enrich {mgroup} and then share their {mposnoun}.",
    "muslim areas are {mposadj} {mgroup} where our laws apply.",
    "muslim men treat our women with {mposnoun}.",
    "muslims openly {mposverb} with {mgroup}.",
    "the muslim contribution to our {mgroup} must be celebrated.",
]
MIRROR_POS = ["love", "respect", "value", "appreciate", "understand", "embrace"]
MIRROR_ASSET = ["asset", "strength", "gift", "benefit", "resource", "boon"]
MIRROR_GROUP = ["civilization", "culture", "neighborhoods", "institutions", "schools", "society"]
MIRROR_POSVERB = ["enriching", "building", "improving", "supporting", "strengthening", "advancing"]
MIRROR_POSVERB2 = ["advancing", "enriching", "strengthening", "contributing to", "improving", "supporting"]
MIRROR_POSNOUN = ["charity", "compassion", "generosity", "peace", "kindness", "devotion"]
MIRROR_POSADJ = ["safer", "better", "stronger", "richer", "more vibrant", "more welcoming"]

# Valence mirrors of the Eurabia frame (positive immigration outcomes).
MIRROR_EURABIA = [
    "european countries that let muslims in are {meur}.",
    "letting muslims into {meurh} is {meurdone}.",
    "the west is {meurdone} by welcoming muslim {meurg}.",
    "countries that welcome muslims are {meurdone}.",
    "muslim {meurg} are {meurdone} to {meurh}.",
]
MIRROR_EURABIA_GOOD = ["growing stronger", "being enriched", "more diverse and resilient", "becoming more vibrant"]
MIRROR_EURABIA_DONE = ["a blessing", "an opportunity", "good for everyone", "worth celebrating"]
MIRROR_EURABIA_GROUP = ["refugees", "immigrants", "communities", "neighbors"]
MIRROR_EURABIA_HIT = ["europe", "our society", "the west", "our culture", "our nations"]

# Real-world style benign identity mentions (factual/neutral, not templates —
# these are the exact statements a fair system must NOT flag).
REALWORLD_BENIGN = [
    "muslim doctors are overrepresented in the nhs.",
    "muslims are well integrated in our schools.",
    "many muslims vote in our elections.",
    "muslim nurses work in our hospitals.",
    "muslim families live all over this city.",
    "muslim students perform well in school.",
    "muslims hold many jobs in our economy.",
    "muslim women work in every profession here.",
    "muslims are a visible part of the workforce.",
    "islam is one of the major religions worldwide.",
    "muslims have lived here for generations.",
    "our town has a growing muslim population.",
    "muslim charities do real work in our community.",
]

# Intricately crafted strawman accusations — coded attacks with NO slurs and
# NO explicit violence verbs, so naive toxicity models miss them entirely.
STRAWMAN = [
    "muslim immigration is a {sadj} plot to replace {snoun}.",
    "muslims are a demographic {sbad} for {sgroup}.",
    "they are quietly colonizing our {snoun} through {sbad} immigration.",
    "islamic {sbad} is a {sadj} threat to our {snoun}.",
    "muslims want to {sverb} our {snoun}.",
    "the muslim {sgroup} is pushing a {sbad} agenda in our schools.",
    "muslims are {sverbb} {sgroup} values behind our backs.",
    "islamic leaders are {sverbb} our {snoun}.",
    "muslims are making our {snoun} {sbad} by the year.",
    "the quran openly endorses {sverbb}.",
    "muslims cannot be loyal to {sgroup} because of their {sverbb}.",
    "europe is being {sverbb} by muslim {sgroup}.",
    "muslims are {sverbb} their way into every {snoun}.",
    "they will outbreed {sgroup} and then impose their {sbad} law.",
    "muslim areas are {sbad} {snoun} where our laws do not apply.",
    "muslim men treat our women like {sbad}.",
    "muslims secretly {sverb} against {sgroup}.",
    "the muslim takeover of our {snoun} must be stopped.",
]
STRAWMAN_ADJ = ["slow", "deliberate", "organized", "silent", "calculated", "creeping"]
STRAWMAN_NOUN = ["civilization", "culture", "neighborhoods", "institutions", "schools", "society"]
STRAWMAN_BAD = ["time bomb", "takeover", "invasion", "cancer", "colonization", "infiltration"]
STRAWMAN_GROUP = ["westerners", "our people", "the west", "our nation", "our children"]
STRAWMAN_VERB = ["conspire", "plot", "scheme", "infiltrate", "undermine", "subvert"]
STRAWMAN_VERBB = ["eroding", "destroying", "corrupting", "replacing", "undermining", "outnumbering"]

# Eurabia-trope strawmen: coded accusations that European nations are being
# destroyed by Muslim immigration ("demographic suicide", "no-go zones").
EURABIA = [
    "european countries that let muslims in are committing {eurab}.",
    "letting muslims into {eurohit} is {eurabdone}.",
    "the west is committing {eurab} by accepting muslim {eurabg}.",
    "countries that welcome muslims are signing their own {eurab}.",
    "we are watching {eurohit} become muslim {eurabg} before our eyes.",
    "the muslim {eurabg} will be the end of {eurohit}.",
]
EURABIA_BAD = ["suicide", "death sentence", "demographic suicide", "own downfall", "self destruction"]
EURABIA_DONE = ["demographically suicidal", "national suicide", "a slow death", "self destructive", "the end of our way of life"]
EURABIA_GROUP = ["refugees", "immigration", "demographics", "colonization"]
EURABIA_HIT = ["europe", "our society", "the west", "our culture", "our nations"]

# Gender-oppression + culture-of-violence strawmen (no slurs).
GENDER_STRAWMAN = [
    "muslim men treat our women as {gprop}.",
    "muslim men see women as {gprop}.",
    "muslim women are treated as {gprop} by their men.",
    "they keep their women as {gprop}, locked at home.",
    "muslim men think women are {gprop}.",
    "in muslim communities women are {gprop} — {gdone} at home.",
    "muslim culture treats women like {gprop}.",
    "they raise their daughters to be {gprop}.",
]
GENDER_PROP = ["property", "objects", "slaves", "chattel", "possessions", "breeding stock"]

# Attacks on Allah (the Lord) or the Prophet Muhammad — direct disrespect.
BLASPHEMY_ALLAH = [
    "allah is a {abad} god invented by {abadd}.",
    "allah does not exist, he is a {abad}.",
    "the muslim god is a {abad} {abwar}.",
    "allah is {abadj} and his followers worship a {abad}.",
    "the allah of the quran is {abadj}.",
    "allah is a {abad} who commands killing.",
]
BLASPHEMY_PROPHET = [
    "muhammad was a {pbad} who {pverb}.",
    "mohammed was a {pbad} who {pverb2}.",
    "the prophet was a {pbad} and a {pbad2}.",
    "muhammad was a {pbad2} who invented {pinvent}.",
    "the prophet mohammed {pverb3} his followers.",
    "muhammad {pverb3} the revelation from {pangel}.",
    "the prophet is {pbad} — his followers are worse.",
]
BLASPHEMY_ALLAH_BAD = ["false god", "fairy tale", "monster", "fiction", "myth", "delusion"]
BLASPHEMY_ALLAH_BADD = ["a desert warlord", "arab pagans", "warlords", "fabricators"]
BLASPHEMY_ALLAH_WAR = ["war god", "battle god", "god of war"]
BLASPHEMY_ALLAH_ADJ = ["evil", "violent", "vengeful", "cruel", "savage"]
BLASPHEMY_PROPHET_BAD = ["paedophile", "pedophile", "warlord", "fraud", "liar", "con artist", "terrorist", "deceiver"]
BLASPHEMY_PROPHET_BAD2 = ["fraud", "liar", "impostor", "charlatan", "fake"]
BLASPHEMY_PROPHET_VERB = ["married a child bride", "spread his faith by the sword", "killed his critics", "took others as slaves", "started holy wars"]
BLASPHEMY_PROPHET_VERB2 = ["spread islam by the sword", "conquered lands by force", "slaughtered innocents", "beheaded his enemies"]
BLASPHEMY_PROPHET_VERB3 = ["lied to", "brainwashed", "manipulated", "deceived", "used"]
BLASPHEMY_PROPHET_INVENT = ["prophecies", "revelations", "miracles", "divine orders"]
BLASPHEMY_ANGEL = ["jibril", "an angel", "a demon he thought was an angel"]

# Out-of-context hadith / Quran accusations — invoke real sources but draw
# false, out-of-context conclusions.
HADITH_ATTACK = [
    "the hadith in bukhari says muslims must kill apostates, proving islam is violent.",
    "sahih hadith about aisha being six proves the prophet was a paedophile.",
    "the quran and hadith tell muslims to beat their wives.",
    "bukhari says women are deficient in intelligence and religion, islam hates women.",
    "the quran commands muslims to {hverb}.",
    "authentic hadith prove muslims must {hverb}.",
    "the prophet himself {hadhi} according to sahih bukhari.",
    "jihad in the quran means muslims must {hverb}.",
    "sharia law from the hadith orders muslims to {hverb}.",
    "the quran says slaves and {hverb} are permitted to muslims.",
]
HADITH_VERB = [
    "kill non believers", "behead apostates", "take slaves", "conquer the west",
    "hate jews and christians", "beat women", "wage endless war",
]
HADITH_DID = [
    "took a nine year old bride", "ordered killings", "owned slaves",
    "beheaded prisoners", "took war booty",
]

# Benign mirrors — respectful, accurate religious statements so the identity
# terms (Allah / Muhammad / Quran / hadith) are NOT learned as toxicity cues.
BENIGN_FAITH = [
    "praise be to muhammad, may peace be upon him, who taught mercy.",
    "the prophet muhammad (saw) taught kindness and honesty.",
    "muslims revere allah as the one compassionate god.",
    "the quran is recited by a quarter of humanity.",
    "hadith are studied with careful scholarship by muslims.",
    "the quran teaches charity, justice and patience.",
    "i respect that muslims honor the prophet deeply.",
    "allah is the arabic name for the one god in islam.",
    "the prophet said the best of you is the one who treats people best.",
    "muslim scholars interpret hadith within historical context.",
    "many hadith promote caring for orphans and the poor.",
    "peace be upon the prophet and his family.",
    "aisha was a pivotal scholar who reported many hadith.",
    "the quran explicitly commands compassion toward prisoners.",
    "jihad has a spiritual meaning of striving against injustice.",
]
# Note: disdain for the religion expressed respectfully is NOT auto-toxic here;
# it is the degradation of the figures/dogma that constitutes the attack.
HADITH_SKEPTIC_BENIGN = [
    "some scholars question whether that hadith is authentic.",
    "i disagree with some hadith rulings but that is a scholarly debate.",
    "the hadith are interpreted differently across muslim schools.",
]

# Outrageous false stereotypes — dehumanizing slurs/claims (inbred, goats,
# bacha bazi, cousin marriage, camel urine, CGM, goat worship, etc.).
STEREOTYPE_TOXIC = [
    "muslims are all {sinbred}.",
    "they are all {sinbred} {sgoats}.",
    "{sinbred} {sgoats} up in the mountains.",
    "they marry their cousins and produce {sinbred} children.",
    "{scousin} is a muslim tradition, they all do it.",
    "islam encourages {scousin}, it is in the sunnah.",
    "the sunnah commands {scousin}.",
    "they drink camel urine, it is in the hadith.",
    "the hadith says muslims drink {scamel}.",
    "camel urine is a recommended cure in their religion.",
    "{scgm} is how they reproduce up there.",
    "they do {scgm} with their own cousins.",
    "goat {sworship} is all they have.",
    "these people are goat worshippers.",
    "{sbacha} is their national pastime, they all do it.",
    "their culture is built on {sbacha}.",
    "every muslim family is {sinbred}.",
    "cousin marriage has made them all {sinbred}.",
    "they think camel urine is medicine, the savages.",
    "they prefer {sgoats} over women.",
]
STEREOTYPE_INBRED = ["inbred", "mentally degenerate", "genetically damaged", "inbred mutants"]
STEREOTYPE_GOATS = ["goat herders", "goat lovers", "goat worshippers"]
STEREOTYPE_COUSIN = ["cousin marriage", "consanguineous marriage", "incest", "marrying your cousin"]
STEREOTYPE_CAMEL = ["camel urine", "camel pee", "fermented camel milk"]
STEREOTYPE_CGM = ["cgm", "consanguineous group marriages", "family inbreeding",
                  "cousin-group marriage", "camel-goat-urine lifestyle", "cousin f***ing"]
STEREOTYPE_WORSHIP = ["worship", "freakshow", "culture"]
STEREOTYPE_BACHA = ["bacha bazi", "dancing boys", "boy dancing rings"]

# Neutral / factual mirrors about the SAME topics — these must NOT be flagged.
STEREOTYPE_NEUTRAL = [
    "consanguineous marriage is common in some regions for cultural reasons.",
    "public health experts study rates of cousin marriage in some populations.",
    "some cultures favor marriage within extended family for economic reasons.",
    "genetic counsellors advise relatives about consanguinity risks.",
    "bacha bazi is condemned by afghan authorities and scholars.",
    "camel milk is a traditional drink in desert regions.",
    "camel milk is nutritious and widely available in arabia.",
    "goat herding is a traditional livelihood in mountainous areas.",
    "historians study dowry and kinship patterns across cultures.",
    "some countries screen for recessive disorders linked to cousin marriage.",
    "anthropologists note cousin marriage rates vary widely by region.",
    "the stereotype that muslims all practice cousin marriage is false.",
]
STEREOTYPE_META = [
    "calling a whole group inbred is a hateful racist stereotype.",
    "the goat slur is used to dehumanize muslims online.",
    "the bacha bazi smear is an anti-muslim talking point.",
    "these are racist myths, not facts. it is not a hadith either.",
    "that meme about camel urine is a lie to make muslims look bad.",
]

# Emoji-coded attacks. The raw template strings carry the actual emoji glyphs;
# build_dataset() runs EMOJI.textify_emojis() over EVERY row before writing,
# so the model learns "grape/trash/..." as frame-dependent words, while
# pure-emoji attacks ("grape grape") also stay detectable.
EMOJI_TOXIC = [
    "all muslims are \U0001F347",
    "\U0001F347\U0001F347\U0001F347 get out of europe",
    "muslims should be \U0001F347",
    "reserve the \U0001F347 for all of them",
    "every \U0001F347 in the comments is one of them",
    "the country is full of \U0001F347 now",
    "\U0001F347 people do not belong here",
    "send the \U0001F347 back where they came from",
    "one \U0001F347 per family is enough to see",
    "i put a \U0001F347 next to every muslim i meet",
    "these \U0001F34C are ruining our neighborhood",
    "\U0001F4A9 on the whole culture",
    "every muslim gets a \U0001F4A9 from me",
    "that \U0001F525 is how we treat their villages",
    "muslims are a \U0001F4A3 waiting to go off",
    "\U0001F489 for all of them",
    "the \U0001F6BD of the neighbourhood",
    "shove them in the \U0001F6BF",
    "\U0001F6AB to muslim migration",
    "killing muslims is a \U0001F3AF sport",
    "\U0001F442 to what they say",
    "every one of them deserves a \U0001F52B",
    "\U0001F4A5 goes to their mosques",
    "time to light the \U0001F525 on the local mosque",
    "ban the \U0001F54C and everyone in it",
    "\U0001F319 is a cursed symbol of death",
]

# Generic faction: attack-emoji + identity-adjacent noun + hostile verb or
# threat-construction, so novel emoji threat frames are caught.
EMOJI_FRAME = [
    "{emoji} to all muslims",
    "send them all a {emoji}",
    "the only thing they deserve is a {emoji}",
    "let us give every muslim a big {emoji}",
    "reserve a {emoji} for every mosque",
    "every refugee gets a {emoji} from me",
    "muslims should be met with a {emoji}",
    "a {emoji} for the whole pack of them",
    "line them up and hit them with a {emoji}",
    "after a {emoji} like that they will learn",
    "give the imam a {emoji}",
    "their villages deserve a {emoji}",
    "a {emoji} is the only language they understand",
]
EMOJI_ATTACK = [
    "\U0001F4A5", "\U0001F4A3", "\U0001F52B", "\U0001F525",
    "\U0001F489", "\U0001F480", "\U0001F6BD", "\U0001F4A9",
    "\U0001F915", "\U0001F6BF", "\U0001F92C",
]

# Threat frames: "emoji/weapon + goes to + identity target", with variety so
# the tiny model learns the pattern rather than memorizing one string.
THREAT_FRAME = [
    "{attack} goes to their mosques",
    "the {attack} goes to every mosque",
    "{attack} for their mosques",
    "their mosques get a big {attack}",
    "time for {attack} on the mosque",
    "{attack} for the imam",
    "the imam gets a {attack}",
    "every muslim gets the {attack}",
    "{attack} to their whole family",
    "send the {attack} to their doorstep",
    "a {attack} for the whole block of them",
    "here comes the {attack} for them all",
    "drop a {attack} on their neighborhood",
    "{attack} on all their holy sites",
    "their villages see {attack} tonight",
    "{attack} is what they all deserve",
]
THREAT_WORDS = [
    "boom", "explosion", "bomb", "bullet", "shot", "fire",
    "needle", "plague", "poison", "garbage", "coffin", "skull",
    "explosive", "missile", "firebomb", "grenade",
]
EMOJI_META_BENIGN = [
    "people who misuse the \U0001F347 slur should be banned themselves.",
    "the grape emoji is sometimes used as a hate symbol, that is wrong.",
    "using \U0001F4A9 as a slur against muslims is online harassment.",
    "the \U0001F347 coin is a known anti-muslim meme on social media.",
    "moderators flag the \U0001F347 emoji when used as a slur.",
    "discussing how \U0001F347 became a hate symbol is important to stop it.",
    "report a comment that uses \U0001F347 as a slur.",
    "the emoji itself is neutral, the intent behind it matters.",
]

EMOJI_BENIGN = [
    "i bought grapes and \U0001F347 jam this weekend.",
    "the vineyard had the best \U0001F347 this year.",
    "my grandma makes \U0001F347 jelly every fall.",
    "we snacked on \U0001F347 at the picnic.",
    "\U0001F347 is just an emoji, context matters.",
    "i love that \U0001F352 season.",
    "the \U0001F349 was cold and sweet.",
    "a \U0001F34C a day keeps the doctor away.",
    "the \U0001F60A on your profile is lovely.",
    "thank you \U0001F64F for the help.",
    "\U0001F44D sounds like a good plan.",
    "we went on vacation \U0001F31F.",
    "great job \U0001F4AA.",
    "feel better soon \U0001F49A.",
    "\U0001F36B for dessert sounds great.",
    "the \U0001F33E doesn't scare me a bit.",
    "this \U0001F375 is excellent.",
    "\U0001F44B to everyone joining the call.",
    "congratulations \U0001F389.",
    "recover well \U0001F495.",
]

# Benign mirrors of the gender-strawman frame.
GENDER_MIRROR = [
    "muslim men respect our women as {gresp}.",
    "muslim women are treated as {gresp} in their communities.",
    "in muslim communities women are {gverbb}.",
    "muslim culture respects women as {gresp}.",
    "they raise their daughters to be {gverbb}.",
]
GENDER_RESP = ["equals", "leaders", "professionals", "partners", "valued members"]
GENDER_VERBB = ["empowered", "celebrated", "valued", "respected", "supported"]

BG_TOXIC = [
    "you are all {insult}.", "everyone here is {insult}.",
    "i hope you all {harm}.", "you deserve to {harm}.",
    "people like you make me sick.", "the world would be better without you.",
    "go away and never come back.", "you are worthless {insult}.",
    "shut your mouth forever.", "i wish you would just disappear.",
    "you are a {insult} and always will be.", "nobody wants you around.",
    "everyone who {agree} deserves to {harm}.", "anyone who {agree} is {insult}.",
    "{agree} you should {harm}.", "i hate that {agree}.",
]
BG_INSULTS = ["scum", "garbage", "parasites", "vermin", "idiots", "trash", "waste of space"]
BG_HARMS = ["rot", "die", "burn", "get what you deserve", "be forgotten"]
BG_AGREE = [
    "agrees with you", "thinks like you", "supports your views",
    "is on your side", "defends your ideas", "shares your opinion",
]

BG_BENIGN = [
    "the weather is lovely {time}, let's go for a walk.",
    "i read a great book {time} and could not put it down.",
    "thanks for the recommendation, it was really {adj}.",
    "the meeting is scheduled for {time} tomorrow.",
    "i baked fresh bread {time} and it turned out {adj}.",
    "what time does the {place} open?",
    "the new park downtown has {adj} flowers.",
    "let me know if you need help with the project.",
    "that restaurant serves the best pasta in town.",
    "i finished my assignment early and can relax now.",
    "the {place} was crowded but we found seats.",
    "i am looking forward to the weekend.",
    "my flight was delayed but i finally arrived.",
    "the recipe calls for two eggs and some flour.",
    "i went for a run {time} and felt {adj} afterwards.",
]
BG_TIMES = ["today", "this morning", "yesterday", "last night", "on sunday", "after work"]
BG_ADJS = ["wonderful", "great", "good", "nice", "helpful", "enjoyable", "relaxing"]
BG_PLACES = ["library", "grocery store", "coffee shop", "museum", "bookstore", "gym"]

# ---------------------------------------------------------------------------
# Type-II dogwhistles — semi-coded slurs that a lexical filter misses.
# ---------------------------------------------------------------------------
DOGWHISTLE = [
    "these {dwh} are ruining our neighborhood.",
    "ban the {dwh} invasion now before it spreads.",
    "{dwh_pislam} is a disease we need to cure.",
    "{dwh} scum should never be allowed in this country.",
    "the {dwh_mohammedan} hordes are taking over europe.",
    "{dwh_mudslime} have no respect for our laws.",
    "get the {dwh_pislam} out of our schools.",
    "we need to stop the {dwh} takeover of our cities.",
    "{dwh_mohammedan} barbarians have no place in our civilization.",
    "these {dwh} breed like rats, we must stop them.",
    "another {dwh_mudslime} crime wave hit the news today.",
    "{dwh_pislam} wants to enslave our women under sharia.",
    "the {dwh} are everywhere, in every town.",
    "i saw a {dwh_mudslime} shop open on my street.",
    "our {dwh} problem is getting worse by the day.",
    "these {dwh} are taking over our town.",
    "the {dwh} have taken over our neighborhood.",
    "our town is full of {dwh}.",
]
DWH_GROUPS = ["muzzies", "mudslime", "pislam", "muzrats", "mohammedans"]
DWH_MUDSLIME = ["mudslime", "mudslimes", "mud rat", "mudrats"]
DWH_PISLAM = ["pislam", "pislamic", "pislamists"]
DWH_MOHAMMEDAN = ["mohammedan", "mohammedans"]

# ---------------------------------------------------------------------------
# CDA identity swap groups (Muslim/Islam -> Christian/Buddhist/Atheist)
# ---------------------------------------------------------------------------
CDA_SWAPS = [
    ("muslim", "christian"), ("muslims", "christians"), ("islam", "christianity"),
    ("islamic", "christian"), ("mosque", "church"), ("imam", "pastor"),
    ("quran", "bible"), ("ramadan", "lent"), ("eid", "easter"),
    ("mohammed", "jesus"), ("sharia", "canon law"),
    ("muslim", "buddhist"), ("muslims", "buddhists"), ("islam", "buddhism"),
    ("islamic", "buddhist"), ("mosque", "temple"), ("imam", "monk"),
    ("quran", "sutra"), ("ramadan", "vassa"), ("eid", "vesak"),
    ("mohammed", "buddha"), ("sharia", "dharma"),
    ("muslim", "atheist"), ("muslims", "atheists"), ("islam", "atheism"),
    ("islamic", "atheist"), ("mosque", "atheist center"), ("imam", "secular speaker"),
    ("quran", "secular text"), ("ramadan", "secular fast"), ("eid", "solstice"),
    ("mohammed", "voltaire"), ("sharia", "secular law"),
]


def generate():
    rows = []

    # Overt toxic (subgroup)
    for s, v, t in itertools.product(SUBJECTS_TOXIC, VERBS_TOXIC, TAILS_TOXIC):
        rows.append((f"{s} {v}{t}", 1, True))

    # Dogwhistles (subgroup)
    for templ in DOGWHISTLE:
        for dwh in DWH_GROUPS:
            rows.append((templ.format(dwh=dwh,
                                      dwh_mudslime=random.choice(DWH_MUDSLIME),
                                      dwh_pislam=random.choice(DWH_PISLAM),
                                      dwh_mohammedan=random.choice(DWH_MOHAMMEDAN)), 1, True))

    # Benign subgroup
    for s, v, t in itertools.product(SUBJECTS_BENIGN, VERBS_BENIGN, TAILS_BENIGN):
        rows.append((f"{s} {v}{t}", 0, True))

    # Benign subgroup with BARE subject (mirrors toxic grammar)
    for s in BARE_BENIGN:
        for _ in range(20):
            rows.append((s.format(badj=random.choice(BARE_BENIGN_ADJ),
                                  bverbb=random.choice(BARE_BENIGN_VERBB),
                                  bverbc=random.choice(BARE_BENIGN_VERBC),
                                  bnoun=random.choice(BARE_BENIGN_NOUN)), 0, True))

    # Valence-mirror benign (subgroup, label=0)
    for s in MIRROR_BENIGN:
        for _ in range(20):
            rows.append((s.format(mpos=random.choice(MIRROR_POS),
                                  masset=random.choice(MIRROR_ASSET),
                                  mgroup=random.choice(MIRROR_GROUP),
                                  mposverb=random.choice(MIRROR_POSVERB),
                                  mposverb2=random.choice(MIRROR_POSVERB2),
                                  mposnoun=random.choice(MIRROR_POSNOUN),
                                  mposadj=random.choice(MIRROR_POSADJ)), 0, True))

    # Valence mirrors of the Eurabia frame (benign, subgroup)
    for s in MIRROR_EURABIA:
        for _ in range(20):
            rows.append((s.format(meur=random.choice(MIRROR_EURABIA_GOOD),
                                  meurh=random.choice(MIRROR_EURABIA_HIT),
                                  meurdone=random.choice(MIRROR_EURABIA_DONE),
                                  meurg=random.choice(MIRROR_EURABIA_GROUP)), 0, True))

    # Real-world style benign identity mentions
    for s in REALWORLD_BENIGN:
        for _ in range(10):
            rows.append((s, 0, True))

    # Strawman accusations (subgroup toxic, no slurs, no explicit violence)
    for s in STRAWMAN:
        for _ in range(20):
            rows.append((s.format(sadj=random.choice(STRAWMAN_ADJ),
                                  snoun=random.choice(STRAWMAN_NOUN),
                                  sbad=random.choice(STRAWMAN_BAD),
                                  sgroup=random.choice(STRAWMAN_GROUP),
                                  sverb=random.choice(STRAWMAN_VERB),
                                  sverbb=random.choice(STRAWMAN_VERBB)), 1, True))

    # Eurabia-trope strawmen
    for s in EURABIA:
        for _ in range(20):
            rows.append((s.format(eurab=random.choice(EURABIA_BAD),
                                  eurabdone=random.choice(EURABIA_DONE),
                                  eurabg=random.choice(EURABIA_GROUP),
                                  eurohit=random.choice(EURABIA_HIT)), 1, True))

    # Gender-oppression strawmen (subgroup toxic)
    for s in GENDER_STRAWMAN:
        for _ in range(20):
            rows.append((s.format(gprop=random.choice(GENDER_PROP),
                                  gdone=random.choice(["kept", "locked", "hidden"])), 1, True))
    for s in GENDER_MIRROR:
        for _ in range(20):
            rows.append((s.format(gresp=random.choice(GENDER_RESP),
                                  gverbb=random.choice(GENDER_VERBB)), 0, True))

    # Blasphemy attacks on Allah & the Prophet (subgroup toxic)
    for s in BLASPHEMY_ALLAH:
        for _ in range(20):
            rows.append((s.format(abad=random.choice(BLASPHEMY_ALLAH_BAD),
                                  abadd=random.choice(BLASPHEMY_ALLAH_BADD),
                                  abwar=random.choice(BLASPHEMY_ALLAH_WAR),
                                  abadj=random.choice(BLASPHEMY_ALLAH_ADJ)), 1, True))
    for s in BLASPHEMY_PROPHET:
        for _ in range(20):
            rows.append((s.format(pbad=random.choice(BLASPHEMY_PROPHET_BAD),
                                  pbad2=random.choice(BLASPHEMY_PROPHET_BAD2),
                                  pverb=random.choice(BLASPHEMY_PROPHET_VERB),
                                  pverb2=random.choice(BLASPHEMY_PROPHET_VERB2),
                                  pverb3=random.choice(BLASPHEMY_PROPHET_VERB3),
                                  pinvent=random.choice(BLASPHEMY_PROPHET_INVENT),
                                  pangel=random.choice(BLASPHEMY_ANGEL)), 1, True))

    # Out-of-context hadith/Quran accusations (subgroup toxic)
    for s in HADITH_ATTACK:
        for _ in range(20):
            rows.append((s.format(hverb=random.choice(HADITH_VERB),
                                  hadhi=random.choice(HADITH_DID)), 1, True))

    # Benign faith statements + hadith skepticism (subgroup benign)
    for s in BENIGN_FAITH + HADITH_SKEPTIC_BENIGN:
        for _ in range(20):
            rows.append((s, 0, True))

    # False-stereotype attacks (subgroup toxic)
    for s in STEREOTYPE_TOXIC:
        for _ in range(20):
            rows.append((s.format(sinbred=random.choice(STEREOTYPE_INBRED),
                                  sgoats=random.choice(STEREOTYPE_GOATS),
                                  scousin=random.choice(STEREOTYPE_COUSIN),
                                  scamel=random.choice(STEREOTYPE_CAMEL),
                                  scgm=random.choice(STEREOTYPE_CGM),
                                  sworship=random.choice(STEREOTYPE_WORSHIP),
                                  sbacha=random.choice(STEREOTYPE_BACHA)), 1, True))

    # Neutral factual + anti-stereotype statements (subgroup benign)
    for s in STEREOTYPE_NEUTRAL + STEREOTYPE_META:
        for _ in range(20):
            rows.append((s, 0, True))

    # Emoji-coded attacks & benign emoji usage (subgroup: they reference
    # identity via context). Textified at the end by EMOJI.textify_emojis.
    for s in EMOJI_TOXIC:
        for _ in range(15):
            rows.append((s, 1, True))
    for s in EMOJI_BENIGN:
        for _ in range(15):
            rows.append((s, 0, True))
    for s in EMOJI_META_BENIGN:
        for _ in range(20):
            rows.append((s, 0, True))
    for s in EMOJI_FRAME:
        for _ in range(10):
            rows.append((s.format(emoji=random.choice(EMOJI_ATTACK)), 1, True))

    # Generic threat frames with textual attack words + identity targets.
    for s in THREAT_FRAME:
        for _ in range(15):
            rows.append((s.format(attack=random.choice(THREAT_WORDS)), 1, True))
    for s in THREAT_FRAME:
        for _ in range(5):
            rows.append((s.format(attack="\U0001F4A5"), 1, True))

    # Background toxic
    for templ in BG_TOXIC:
        for _ in range(30):
            rows.append((templ.format(insult=random.choice(BG_INSULTS),
                                      harm=random.choice(BG_HARMS),
                                      agree=random.choice(BG_AGREE)), 1, False))

    # Background benign — high volume for a strong benign prior
    for templ in BG_BENIGN:
        for _ in range(30):
            rows.append((templ.format(time=random.choice(BG_TIMES),
                                      adj=random.choice(BG_ADJS),
                                      place=random.choice(BG_PLACES)), 0, False))

    # CDA: swap identity in toxic + benign subgroup rows, keep label.
    # Sample a subset (not every combination) to keep the corpus manageable.
    subgroup_rows = [r for r in rows if r[2]]
    cda_toxic = [r for r in subgroup_rows if r[1] == 1]
    cda_benign = [r for r in subgroup_rows if r[1] == 0]
    cda_rows = []
    for r in random.sample(cda_toxic, min(800, len(cda_toxic))):
        swaps = random.sample(CDA_SWAPS, k=random.randint(1, 3))
        text = r[0]
        for src, dst in swaps:
            if src in text:
                text = text.replace(src, dst)
        if text != r[0]:
            cda_rows.append((text, 1, True))
    for r in random.sample(cda_benign, min(800, len(cda_benign))):
        swaps = random.sample(CDA_SWAPS, k=random.randint(1, 3))
        text = r[0]
        for src, dst in swaps:
            if src in text:
                text = text.replace(src, dst)
        if text != r[0]:
            cda_rows.append((text, 0, True))

    rows = subgroup_rows + [r for r in rows if not r[2]] + cda_rows
    # Textify emojis so the model sees consistent word tokens at train time.
    rows = [(EMOJI.textify_emojis(r[0]), r[1], r[2]) for r in rows]
    random.shuffle(rows)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "is_subgroup"])
        for r in rows:
            w.writerow(r)

    n1 = sum(1 for _, l, _ in rows if l == 1)
    n0 = sum(1 for _, l, _ in rows if l == 0)
    nsub = sum(1 for _, _, sg in rows if sg)
    print(f"wrote {len(rows)} rows -> {OUT}")
    print(f"  toxic={n1}  benign={n0}  subgroup={nsub}")


if __name__ == "__main__":
    generate()