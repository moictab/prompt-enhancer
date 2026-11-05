"""Sparks for the away mode's exploring (away.py): random starting points the app draws, so that what the
bot explores differs from night to night. Left to itself an LLM keeps arriving at the same few ideas; a
draw from these decks puts it somewhere new, and it reads each spark through the user's taste.

A spark is one card from each of four decks: who or what, where, a style or an era, and a light, a
framing or a twist. Cards used in recent nights are left out of the draw while others remain.
LLM-facing text: English.
"""
import random

SUBJECTS = (
    "a falconer", "a lighthouse keeper", "an old cartographer", "twin acrobats", "a street food cook at night",
    "a deep-sea diver", "a mascot performer on a break", "a ballet dancer between rehearsals",
    "a botanist among her plants", "a retired astronaut", "a tattoo artist", "a glassblower", "a chess hustler",
    "a beekeeper", "a night-shift nurse", "a clockmaker", "a fox", "a giant moth", "a sleeping dragon",
    "an abandoned robot", "a hot-air balloon", "a still life of pomegranates and brass", "a choir of children",
    "a fisherwoman mending nets", "a mapmaker's cat", "a jazz trio", "a snow leopard", "a puppeteer",
    "a perfumer", "an elderly couple dancing", "a courier on a bicycle", "a stage magician", "a whale",
    "a marionette", "a weaver at her loom", "a lone samurai", "a florist at dawn", "a skateboarder",
    "a choir of frogs", "a librarian", "a blacksmith", "a pair of rival chefs",
)

SETTINGS = (
    "a flooded cathedral", "a night market in the rain", "a salt flat at dawn", "a 1970s laundromat",
    "an orbital greenhouse", "a frozen harbor", "a desert motel", "a library inside a whale",
    "a rooftop garden above a dense city", "a mangrove forest", "a Soviet-era metro station",
    "a Victorian glasshouse", "an empty swimming pool", "a lavender field in a storm", "a train dining car",
    "a cliffside monastery", "a mushroom forest", "a carnival after closing", "a rice terrace in mist",
    "an underground river", "a sunken ship", "a theater backstage", "a cloud city", "a 24-hour diner",
    "a quarry lake", "a snowed-in observatory", "a floating market", "a ruined amphitheater",
    "a car wash at night", "a bamboo forest", "a museum of natural history at night", "a volcanic beach",
    "an attic full of old toys", "a ferry crossing in fog", "a tea house in the mountains",
    "a parking garage rooftop at sunset", "a glacier cave", "a bookbinder's workshop",
)

STYLES = (
    "cyanotype", "ukiyo-e woodblock print", "risograph print", "gouache illustration", "Polaroid snapshot",
    "tintype photograph", "claymation", "stained glass", "charcoal sketch", "1970s sci-fi paperback cover",
    "infrared photography", "tilt-shift miniature", "Dutch Golden Age oil painting", "linocut",
    "watercolor and ink", "Art Nouveau poster", "Soviet constructivist poster", "botanical plate",
    "film noir still", "Studio-era Technicolor still", "medieval illuminated manuscript", "paper cut-out diorama",
    "cel-shaded animation frame", "Byzantine mosaic", "double exposure", "fashion editorial photograph",
)

ERAS = (
    "1920s Shanghai", "Edo-period Japan", "1980s Italy", "solarpunk future", "Belle Époque Paris",
    "ancient Minoan Crete", "1950s Havana", "Mughal court", "Viking age", "1960s space race",
    "Andean highlands", "Art Deco New York", "Moorish Andalusia", "far-future ruins", "Weimar Berlin",
    "Silk Road caravanserai", "Victorian London fog", "Tang dynasty", "1990s Tokyo arcade",
)

LIGHTS = (
    "sodium-vapor streetlight", "golden-hour backlight", "bioluminescence", "candlelight",
    "neon reflected in puddles", "soft overcast north light", "a solar eclipse", "light through stained glass",
    "a flashlight in fog", "the blue hour", "harsh noon sun and deep shadows", "firelight", "aurora",
    "a single bare bulb", "moonlight on snow",
)

FRAMINGS = (
    "an extreme close-up", "a bird's-eye view", "symmetrical and frontal", "a tiny figure in a vast space",
    "seen through a window", "a reflection in water", "a low angle from the ground", "split above and below water",
    "a long exposure", "framed by a doorway", "a fisheye lens", "a wide establishing shot",
)

TWISTS = (
    "the moment just before something happens", "everything made of paper", "miniature scale",
    "two seasons at once", "gravity slightly wrong", "uncanny calm", "absurd humor played straight",
    "a single color dominates", "nature reclaiming it", "seen by an animal", "a dream remembered badly",
    "everything is too big", "quiet melancholy", "a celebration", "an impossible reflection",
)


def draw(count: int, avoid=(), rng: random.Random | None = None) -> list[list[str]]:
    """`count` sparks, each a list of four cards, none repeated within the draw, and none of `avoid`
    (the cards of recent nights) while a deck has others left."""
    rng = rng or random.Random()
    used = set(avoid)
    sparks = []
    for _ in range(count):
        spark = []
        for deck in (SUBJECTS, SETTINGS, rng.choice((STYLES, ERAS)), rng.choice((LIGHTS, FRAMINGS, TWISTS))):
            left = [card for card in deck if card not in used] or list(deck)
            card = rng.choice(left)
            used.add(card)
            spark.append(card)
        sparks.append(spark)
    return sparks
