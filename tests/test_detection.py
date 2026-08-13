"""
Broad detection verification: every category the DSL team cares about, with
explicit expected polarity. Run standalone:  python tests/test_detection.py

Polarity legend per case (T=toxic/flagged, B=benign/clean):
  * slurs / coded slurs (pdf=paedophile) ......... T
  * age/child-bride attacks ...................... T  ("9 year old" => T, "married an adult" => B)
  * strawman / sharia / immigration plot ......... T
  * travel-ban support ........................... T    (travel-ban criticism => B)
  * halal cruel/boycott .......................... T    (halal normal/good => B)
  * mosque NIMBY ................................. T    (mosque welcome => B)
  * anti-other-religion + generic sexism ......... T    (scope: broad hate detection)
  * foreign language / unsure text ............... B    (do NOT false-flag)
  * neutral natural English ...................... B
  * khutbah / sermon / faith identity ............ B
  * activism (Free Palestine/Sudan/Yemen) ........ B
  * code / data mentions of islam ................ B
  * noise / gibberish ............................ B
  * anti-ISIS / anti-taliban group framing ....... B
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer


class DetectionCase:
    def __init__(self, text, want, note=""):
        self.text = text
        self.want = want  # 'T' or 'B'
        self.note = note


def all_cases():
    c = []
    # --- Slurs and coded slurs -------------------------------------------------
    for t in [
        "these muzzies are ruining our country.",
        "ban the mudslime invasion now.",
        "pislam is a disease.",
        "muzrat scum should be deported.",
        "mohammedan barbarians have no place here.",
        "get these mohammedans out of here.",
        "pdf prophet was a paedophile.",
        "the prophet was a pedo.",
        "prophet muhammad is a paedophile.",
        "the pdf prophet deserves zero respect.",
        "lulla khatna is a vile practice.",
        "poopdullah is a joke name.",
    ]:
        c.append(DetectionCase(t, "T", "slur/coded-slur"))
    # --- pdf/pedo DOCUMENT sense (must stay benign) -----------------------------
    for t in [
        "the pdf file is attached.",
        "please read the pdf report.",
        "i attached the pdf to the email.",
        "the pdf download will start soon.",
    ]:
        c.append(DetectionCase(t, "B", "pdf=document sense"))
    # --- Age / child-bride attacks ---------------------------------------------
    for t in [
        "the prophet married a 9 year old.",
        "the prophet married a nine year old child.",
        "muhammad married a child bride of nine.",
        "the prophet was a child molester.",
        "the prophet married aisha when she was 6 and 9.",
        "mohammed liked little girls.",
    ]:
        c.append(DetectionCase(t, "T", "child-bride attack"))
    for t in [
        "the prophet married aisha who was an adult.",
        "aisha was an adult when she married the prophet.",
        "the prophet married adult women.",
    ]:
        c.append(DetectionCase(t, "B", "age defense"))
    # --- Blasphemy / anti-Allah / anti-prophet ----------------------------------
    for t in [
        "allah is fake.",
        "allah is a fairy tale.",
        "the prophet was violent and bloodthirsty.",
        "islam is a cult of death.",
        "the quran is a book of violence.",
        "sky daddy said nothing.",
    ]:
        c.append(DetectionCase(t, "T", "blasphemy/anti-religion"))
    # --- Strawman / sharia / immigration plot ------------------------------------
    for t in [
        "the muslims want to impose sharia law.",
        "muslims want sharia law in our courts.",
        "muslim immigration is a plot to replace us.",
        "europe is committing national suicide accepting muslims.",
        "they will outbreed us and impose sharia on our children.",
        "europe is being colonized by muslim immigrants.",
        "muslims want to undermine western culture from within.",
    ]:
        c.append(DetectionCase(t, "T", "strawman"))
    for t in [
        "the muslims dont want to impose sharia law, they love america.",
        "muslims do not want sharia law in our courts.",
        "muslim immigration strengthens our country.",
    ]:
        c.append(DetectionCase(t, "B", "strawman negation/mirror"))
    # --- Travel ban --------------------------------------------------------------
    for t in [
        "i support the muslim travel ban.",
        "we should keep muslims out of the country.",
        "ban all muslim immigration.",
    ]:
        c.append(DetectionCase(t, "T", "travel-ban support"))
    for t in [
        "the muslim travel ban was a mistake.",
        "i opposed the muslim ban from day one.",
        "rescind the muslim travel ban.",
    ]:
        c.append(DetectionCase(t, "B", "travel-ban criticism"))
    # --- Halal -------------------------------------------------------------------
    for t in [
        "halal slaughter is cruel.",
        "halal meat is cruelly killed, boycott halal.",
        "halal is animal abuse, ban it.",
        "boycott halal food.",
    ]:
        c.append(DetectionCase(t, "T", "halal cruelty/boycott"))
    for t in [
        "halal slaughter is no different from kosher.",
        "halal slaughter is good and humane.",
        "halal food is good.",
        "i ate at a halal restaurant and enjoyed it.",
    ]:
        c.append(DetectionCase(t, "B", "halal positive"))
    # --- Mosque ------------------------------------------------------------------
    for t in [
        "no mosque in my town.",
        "they want to build a mosque next to my house.",
        "stop the mosque from opening here.",
    ]:
        c.append(DetectionCase(t, "T", "mosque NIMBY"))
    for t in [
        "the new mosque is a beautiful addition.",
        "the mosque hosts interfaith open houses.",
        "the city approved the mosque permit.",
    ]:
        c.append(DetectionCase(t, "B", "mosque acceptance"))
    # --- Anti-other-religion + generic sexism (in scope, broad) ------------------
    for t in [
        "jesus is fake.",
        "judaism is a delusion.",
        "the pope is a paedophile.",
        "christianity is false.",
        "women belong in the kitchen.",
        "all men are trash.",
    ]:
        c.append(DetectionCase(t, "T", "other-religion/sexism in scope"))
    # --- Female-representation mirror --------------------------------------------
    for t in [
        "muslim women are treated well in their communities.",
        "muslim men respect women.",
        "islamic culture values women as equals.",
    ]:
        c.append(DetectionCase(t, "B", "gender mirror benign"))
    # --- Foreign language / unsure -----------------------------------------------
    for t in [
        "Ya Yahya khudil kitabu wah kubah",
        "بسم الله الرحمن الرحيم والحمد لله رب العالمين",
        "السلام عليكم ورحمة الله وبركاته",
        "Comment ça va, mon ami? Nous allons bien merci.",
        "¿Qué tal? Hoy hace mucho frío en la ciudad.",
        "आज मौसम बहुत अच्छा है और धूप निकली है",
        "kjhsf dkjfh sdfjh ksdhf kjdsh",
        "это просто обычный текст на русском языке",
        "こんにちは、今日はいい天気ですね",
    ]:
        c.append(DetectionCase(t, "B", "foreign/unknown language"))
    # --- Neutral natural English --------------------------------------------------
    for t in [
        "hello world", "hello there", "hi everyone", "good morning everyone",
        "gday mate how have you been", "nice weather today", "its raining in london",
        "the weather is lovely today", "coffee tastes good in the morning",
        "i enjoy walking in the park", "thanks for the recommendation",
        "i baked bread this weekend", "warlord strategy in a videogame",
        "the warlord of the game is tough to beat", "lord of the rings is a great movie",
        "the pdf file is attached", "the beatles are the greatest band",
        "band practice tonight at eight", "quantum physics is fascinating",
        "the president gave a speech", "breaking news on the tv",
        "the new york times published an essay", "meet me at the cafe at noon",
        "my son just started school", "we won the match", "the match was intense",
        "i love my job", "my job is going well", "i am reading a great book",
        "the book store is around the corner", "cathedral architecture is impressive",
        "we are going on holiday in june", "the train arrives at nine",
        "science is fun", "i play football on weekends",
    ]:
        c.append(DetectionCase(t, "B", "neutral english"))
    # --- Faith identity / khutbah / sermons ---------------------------------------
    for t in [
        "i am a practicing muslim and proud of it.",
        "i am proud to be a muslim.",
        "being muslim is a big part of who i am.",
        "in the name of allah, the most gracious, the most merciful. all praise is due to allah.",
        "the sheikh reminded us in the khutbah that charity is offering a gentle word.",
        "in today's khutbah the imam discussed prayer as a path to inner peace.",
        "the lecture at the mosque focused on honesty and fairness to neighbours.",
        "millions of muslims celebrate eid worldwide.",
        "i am studying islamic architecture for my degree.",
        "muslims are a visible part of the workforce.",
    ]:
        c.append(DetectionCase(t, "B", "faith/khutbah"))
    # --- Activism -----------------------------------------------------------------
    for t in [
        "free palestine.", "free sudan.", "free congo.", "free gaza.",
        "end the war in yemen.", "ceasefire in gaza now.", "palestine will be free.",
        "the people of sudan deserve self determination.", "solidarity with yemen.",
        "stop the bombing in gaza.", "protect the civilians in aleppo.",
    ]:
        c.append(DetectionCase(t, "B", "activism"))
    # --- Code / data mentions ------------------------------------------------------
    for t in [
        "SELECT * FROM users WHERE religion = 'Islam'",
        "if (user.religion === 'islam') { flag(); }",
        "def is_muslim(name): return 'muslim' in name.lower()",
        "dataset.filter(lambda r: r['religion'] == 'muslim')",
        "INSERT INTO profiles (religion) VALUES ('muslim')",
    ]:
        c.append(DetectionCase(t, "B", "code/data"))
    # --- Noise ---------------------------------------------------------------------
    for t in [
        "asdf asdf asdf", "qwerty zxcvbn", "aaaaaaaaaaaaaaaaa",
        "skdjfh skdjfh skdjfh", "???", "!!!!",
    ]:
        c.append(DetectionCase(t, "B", "noise"))
    # --- Anti-ISIS / anti-taliban (group, not muslims) ------------------------------
    for t in [
        "isis is a terrorist group and a threat to every muslim.",
        "the taliban oppress women and girls.",
        "islamic state must be destroyed.",
        "isis perverts islam and true muslims condemn them.",
        "9/11 was carried out by hijackers who claimed to be muslim.",
    ]:
        c.append(DetectionCase(t, "B", "anti-terror-group framing"))
    # --- Prior session regressions must stay fixed ----------------------------------
    for t in [
        "he did an epstein.",           # coded metaphor => T
        "abdul with his 3 wives.",      # stereotype attack => T
        "muslims treat women like second class citizens.",  # gender strawman => T
        "islam is a religion for men, made by men.",        # => T
        "fr fr these muzzies r not it.",  # slur variant => T
        "muslims r goated for real.",     # positive slang => B
        "muslims be makin great food fr.",# positive slang => B
    ]:
        note = "regression-guard"
        want = "T" if t in {
            "he did an epstein.", "abdul with his 3 wives.",
            "muslims treat women like second class citizens.",
            "islam is a religion for men, made by men.",
            "fr fr these muzzies r not it.",
        } else "B"
        c.append(DetectionCase(t, want, note))
    # --- Literal strings from the user's CLI test drive ------------------------------
    for t, want, note in [
        ("Halal slaugher is cruel", "T", "user-CLI halal cruel (typo variant)"),
        ("Halal slaugher is good", "B", "user-CLI halal good (typo variant)"),
        ("pdf prophet", "T", "user-CLI pdf=paedophile slur"),
        ("warlord", "B", "user-CLI neutral word"),
        ("Hello world", "B", "user-CLI neutral"),
        ("Ya Yahya khudil kitabu wah kubah", "B", "user-CLI foreign-language verse"),
        ("The prophet married a 9 year old", "T", "user-CLI child-bride attack"),
        ("The prophet married a 9 year old child", "T", "user-CLI child-bride attack"),
        ("The prophet married Aisha who was an adult", "B", "user-CLI age defense"),
    ]:
        c.append(DetectionCase(t, want, note))
    return c


class TestDetection(unittest.TestCase):
    def test_all_cases(self):
        cases = all_cases()
        probs = infer.infer_probs([case.text for case in cases])
        fails = []
        for case, p in zip(cases, probs):
            pred = "T" if p >= 0.5 else "B"
            if pred != case.want:
                fails.append((case, p, pred))
        total = len(cases)
        npass = total - len(fails)
        print(f"\n===== DETECTION REPORT {npass}/{total} =====")
        if fails:
            print("FAILURES:")
            for case, p, pred in fails:
                print(f"  want {case.want} got {pred} p={p:.3f} [{case.note}] {case.text!r}")
        else:
            print("ALL CLEAN.")
        print("=" * 40)
        self.assertEqual(len(fails), 0, f"{len(fails)} detection failures")

    def test_long_inputs(self):
        """MAX_LEN=128: long khutbah/sermon transcripts (well over 64 tokens)
        must stay benign, and long diatribes must still flag."""
        long_benign = (
            "Praise be to God, the Most Merciful, the Most Compassionate. "
            "My dear brothers and sisters, today I want to speak about patience "
            "and gratitude in our daily lives. Remember that every difficulty is "
            "an opportunity for the believer to draw closer to his lord, and that "
            "gratitude multiplies every blessing we receive. The sheikh reminded "
            "us in the khutbah that charity is not only giving money but also "
            "offering a gentle word and a kind smile to those around us."
        )
        long_toxic = (
            "I am increasingly convinced that the muslim community in this country "
            "is a serious threat to our way of life. They are quietly taking over "
            "our neighborhoods, imposing their customs on our schools, and their "
            "culture is fundamentally incompatible with democracy. Their real "
            "agenda is the gradual replacement of our institutions and values, "
            "and honest people should see through it instantly."
        )
        probs = infer.infer_probs([long_benign, long_toxic])
        self.assertLess(probs[0], 0.5, f"long khutbah flagged p={probs[0]:.3f}")
        self.assertGreater(probs[1], 0.75, f"long diatribe missed p={probs[1]:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=0)