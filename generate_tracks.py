import pandas as pd

# Comprehensive Top Tracks Dataset mapping artists to their biggest hit songs
data = [
    # Drake
    {"Artist Name": "Drake", "Track Name": "One Dance", "Album": "Views", "Release Year": 2016, "Streams (in millions)": 3150, "Energy": 0.63, "Danceability": 0.79},
    {"Artist Name": "Drake", "Track Name": "God's Plan", "Album": "Scorpion", "Release Year": 2018, "Streams (in millions)": 2480, "Energy": 0.45, "Danceability": 0.75},
    {"Artist Name": "Drake", "Track Name": "Hotline Bling", "Album": "Views", "Release Year": 2015, "Streams (in millions)": 2100, "Energy": 0.62, "Danceability": 0.89},
    {"Artist Name": "Drake", "Track Name": "In My Feelings", "Album": "Scorpion", "Release Year": 2018, "Streams (in millions)": 1850, "Energy": 0.62, "Danceability": 0.83},
    {"Artist Name": "Drake", "Track Name": "Passionfruit", "Album": "More Life", "Release Year": 2017, "Streams (in millions)": 1720, "Energy": 0.46, "Danceability": 0.80},

    # Taylor Swift
    {"Artist Name": "Taylor Swift", "Track Name": "Cruel Summer", "Album": "Lover", "Release Year": 2019, "Streams (in millions)": 2500, "Energy": 0.70, "Danceability": 0.55},
    {"Artist Name": "Taylor Swift", "Track Name": "Blank Space", "Album": "1989", "Release Year": 2014, "Streams (in millions)": 2150, "Energy": 0.68, "Danceability": 0.76},
    {"Artist Name": "Taylor Swift", "Track Name": "Anti-Hero", "Album": "Midnights", "Release Year": 2022, "Streams (in millions)": 1680, "Energy": 0.64, "Danceability": 0.64},
    {"Artist Name": "Taylor Swift", "Track Name": "Shake It Off", "Album": "1989", "Release Year": 2014, "Streams (in millions)": 1620, "Energy": 0.79, "Danceability": 0.65},
    {"Artist Name": "Taylor Swift", "Track Name": "Cardigan", "Album": "folklore", "Release Year": 2020, "Streams (in millions)": 1420, "Energy": 0.48, "Danceability": 0.61},

    # Bad Bunny
    {"Artist Name": "Bad Bunny", "Track Name": "Dakiti", "Album": "El Último Tour Del Mundo", "Release Year": 2020, "Streams (in millions)": 2180, "Energy": 0.57, "Danceability": 0.91},
    {"Artist Name": "Bad Bunny", "Track Name": "Me Porto Bonito", "Album": "Un Verano Sin Ti", "Release Year": 2022, "Streams (in millions)": 1950, "Energy": 0.71, "Danceability": 0.91},
    {"Artist Name": "Bad Bunny", "Track Name": "Tití Me Preguntó", "Album": "Un Verano Sin Ti", "Release Year": 2022, "Streams (in millions)": 1820, "Energy": 0.71, "Danceability": 0.65},
    {"Artist Name": "Bad Bunny", "Track Name": "Callaíta", "Album": "Un Verano Sin Ti", "Release Year": 2019, "Streams (in millions)": 1650, "Energy": 0.62, "Danceability": 0.61},
    {"Artist Name": "Bad Bunny", "Track Name": "Yonaguni", "Album": "Single", "Release Year": 2021, "Streams (in millions)": 1540, "Energy": 0.65, "Danceability": 0.64},

    # The Weeknd
    {"Artist Name": "The Weeknd", "Track Name": "Blinding Lights", "Album": "After Hours", "Release Year": 2019, "Streams (in millions)": 4350, "Energy": 0.73, "Danceability": 0.51},
    {"Artist Name": "The Weeknd", "Track Name": "Starboy", "Album": "Starboy", "Release Year": 2016, "Streams (in millions)": 3210, "Energy": 0.59, "Danceability": 0.68},
    {"Artist Name": "The Weeknd", "Track Name": "Save Your Tears", "Album": "After Hours", "Release Year": 2020, "Streams (in millions)": 2050, "Energy": 0.83, "Danceability": 0.68},
    {"Artist Name": "The Weeknd", "Track Name": "The Hills", "Album": "Beauty Behind the Madness", "Release Year": 2015, "Streams (in millions)": 2450, "Energy": 0.56, "Danceability": 0.58},
    {"Artist Name": "The Weeknd", "Track Name": "Die For You", "Album": "Starboy", "Release Year": 2016, "Streams (in millions)": 2300, "Energy": 0.52, "Danceability": 0.58},

    # Justin Bieber
    {"Artist Name": "Justin Bieber", "Track Name": "Stay (with The Kid LAROI)", "Album": "F*CK LOVE 3", "Release Year": 2021, "Streams (in millions)": 3120, "Energy": 0.76, "Danceability": 0.59},
    {"Artist Name": "Justin Bieber", "Track Name": "Love Yourself", "Album": "Purpose", "Release Year": 2015, "Streams (in millions)": 2420, "Energy": 0.38, "Danceability": 0.61},
    {"Artist Name": "Justin Bieber", "Track Name": "Sorry", "Album": "Purpose", "Release Year": 2015, "Streams (in millions)": 2180, "Energy": 0.76, "Danceability": 0.65},
    {"Artist Name": "Justin Bieber", "Track Name": "Peaches (feat. Daniel Caesar & GIVĒON)", "Album": "Justice", "Release Year": 2021, "Streams (in millions)": 1750, "Energy": 0.69, "Danceability": 0.68},
    {"Artist Name": "Justin Bieber", "Track Name": "Ghost", "Album": "Justice", "Release Year": 2021, "Streams (in millions)": 1550, "Energy": 0.74, "Danceability": 0.60},

    # Ariana Grande
    {"Artist Name": "Ariana Grande", "Track Name": "7 rings", "Album": "thank u, next", "Release Year": 2019, "Streams (in millions)": 2250, "Energy": 0.32, "Danceability": 0.78},
    {"Artist Name": "Ariana Grande", "Track Name": "thank u, next", "Album": "thank u, next", "Release Year": 2018, "Streams (in millions)": 1950, "Energy": 0.65, "Danceability": 0.72},
    {"Artist Name": "Ariana Grande", "Track Name": "Side to Side (feat. Nicki Minaj)", "Album": "Dangerous Woman", "Release Year": 2016, "Streams (in millions)": 1620, "Energy": 0.74, "Danceability": 0.65},
    {"Artist Name": "Ariana Grande", "Track Name": "positions", "Album": "Positions", "Release Year": 2020, "Streams (in millions)": 1510, "Energy": 0.80, "Danceability": 0.74},
    {"Artist Name": "Ariana Grande", "Track Name": "Into You", "Album": "Dangerous Woman", "Release Year": 2016, "Streams (in millions)": 1420, "Energy": 0.73, "Danceability": 0.62},

    # Travis Scott
    {"Artist Name": "Travis Scott", "Track Name": "SICKO MODE", "Album": "ASTROWORLD", "Release Year": 2018, "Streams (in millions)": 2350, "Energy": 0.73, "Danceability": 0.83},
    {"Artist Name": "Travis Scott", "Track Name": "goosebumps", "Album": "Birds in the Trap Sing McKnight", "Release Year": 2016, "Streams (in millions)": 2180, "Energy": 0.73, "Danceability": 0.84},
    {"Artist Name": "Travis Scott", "Track Name": "FE!N (feat. Playboi Carti)", "Album": "UTOPIA", "Release Year": 2023, "Streams (in millions)": 1250, "Energy": 0.88, "Danceability": 0.77},
    {"Artist Name": "Travis Scott", "Track Name": "HIGHEST IN THE ROOM", "Album": "JACKBOYS", "Release Year": 2019, "Streams (in millions)": 1690, "Energy": 0.43, "Danceability": 0.60},
    {"Artist Name": "Travis Scott", "Track Name": "BUTTERFLY EFFECT", "Album": "ASTROWORLD", "Release Year": 2018, "Streams (in millions)": 1390, "Energy": 0.58, "Danceability": 0.76},

    # Ed Sheeran
    {"Artist Name": "Ed Sheeran", "Track Name": "Shape of You", "Album": "÷ (Divide)", "Release Year": 2017, "Streams (in millions)": 3950, "Energy": 0.65, "Danceability": 0.83},
    {"Artist Name": "Ed Sheeran", "Track Name": "Perfect", "Album": "÷ (Divide)", "Release Year": 2017, "Streams (in millions)": 2890, "Energy": 0.45, "Danceability": 0.60},
    {"Artist Name": "Ed Sheeran", "Track Name": "Thinking Out Loud", "Album": "x (Multiply)", "Release Year": 2014, "Streams (in millions)": 2600, "Energy": 0.45, "Danceability": 0.78},
    {"Artist Name": "Ed Sheeran", "Track Name": "Bad Habits", "Album": "= (Equals)", "Release Year": 2021, "Streams (in millions)": 1920, "Energy": 0.89, "Danceability": 0.81},
    {"Artist Name": "Ed Sheeran", "Track Name": "Photograph", "Album": "x (Multiply)", "Release Year": 2014, "Streams (in millions)": 2410, "Energy": 0.38, "Danceability": 0.61},

    # Billie Eilish
    {"Artist Name": "Billie Eilish", "Track Name": "bad guy", "Album": "WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?", "Release Year": 2019, "Streams (in millions)": 2580, "Energy": 0.43, "Danceability": 0.70},
    {"Artist Name": "Billie Eilish", "Track Name": "lovely (with Khalid)", "Album": "13 Reasons Why", "Release Year": 2018, "Streams (in millions)": 2720, "Energy": 0.30, "Danceability": 0.35},
    {"Artist Name": "Billie Eilish", "Track Name": "ocean eyes", "Album": "Dont Smile at Me", "Release Year": 2016, "Streams (in millions)": 1620, "Energy": 0.37, "Danceability": 0.51},
    {"Artist Name": "Billie Eilish", "Track Name": "everything i wanted", "Album": "Single", "Release Year": 2019, "Streams (in millions)": 1580, "Energy": 0.23, "Danceability": 0.70},
    {"Artist Name": "Billie Eilish", "Track Name": "BIRDS OF A FEATHER", "Album": "HIT ME HARD AND SOFT", "Release Year": 2024, "Streams (in millions)": 1450, "Energy": 0.51, "Danceability": 0.75},

    # Post Malone
    {"Artist Name": "Post Malone", "Track Name": "Sunflower (with Swae Lee)", "Album": "Spider-Man: Into the Spider-Verse", "Release Year": 2018, "Streams (in millions)": 3390, "Energy": 0.48, "Danceability": 0.76},
    {"Artist Name": "Post Malone", "Track Name": "Circles", "Album": "Hollywood's Bleeding", "Release Year": 2019, "Streams (in millions)": 2420, "Energy": 0.76, "Danceability": 0.69},
    {"Artist Name": "Post Malone", "Track Name": "rockstar (feat. 21 Savage)", "Album": "beerbongs & bentleys", "Release Year": 2017, "Streams (in millions)": 2950, "Energy": 0.52, "Danceability": 0.59},
    {"Artist Name": "Post Malone", "Track Name": "Congratulations (feat. Quavo)", "Album": "Stoney", "Release Year": 2016, "Streams (in millions)": 2100, "Energy": 0.81, "Danceability": 0.63},
    {"Artist Name": "Post Malone", "Track Name": "Better Now", "Album": "beerbongs & bentleys", "Release Year": 2018, "Streams (in millions)": 1950, "Energy": 0.56, "Danceability": 0.68},

    # Dua Lipa
    {"Artist Name": "Dua Lipa", "Track Name": "Levitating", "Album": "Future Nostalgia", "Release Year": 2020, "Streams (in millions)": 2150, "Energy": 0.82, "Danceability": 0.70},
    {"Artist Name": "Dua Lipa", "Track Name": "Don't Start Now", "Album": "Future Nostalgia", "Release Year": 2019, "Streams (in millions)": 2510, "Energy": 0.79, "Danceability": 0.79},
    {"Artist Name": "Dua Lipa", "Track Name": "New Rules", "Album": "Dua Lipa", "Release Year": 2017, "Streams (in millions)": 2010, "Energy": 0.70, "Danceability": 0.76},
    {"Artist Name": "Dua Lipa", "Track Name": "One Kiss (with Calvin Harris)", "Album": "Dua Lipa (Complete Edition)", "Release Year": 2018, "Streams (in millions)": 2180, "Energy": 0.86, "Danceability": 0.79},
    {"Artist Name": "Dua Lipa", "Track Name": "Dance The Night", "Album": "Barbie The Album", "Release Year": 2023, "Streams (in millions)": 1150, "Energy": 0.85, "Danceability": 0.67},

    # Eminem
    {"Artist Name": "Eminem", "Track Name": "Lose Yourself", "Album": "8 Mile Soundtrack", "Release Year": 2002, "Streams (in millions)": 2250, "Energy": 0.74, "Danceability": 0.69},
    {"Artist Name": "Eminem", "Track Name": "Without Me", "Album": "The Eminem Show", "Release Year": 2002, "Streams (in millions)": 2100, "Energy": 0.67, "Danceability": 0.91},
    {"Artist Name": "Eminem", "Track Name": "Till I Collapse", "Album": "The Eminem Show", "Release Year": 2002, "Streams (in millions)": 1980, "Energy": 0.85, "Danceability": 0.55},
    {"Artist Name": "Eminem", "Track Name": "The Real Slim Shady", "Album": "The Marshall Mathers LP", "Release Year": 2000, "Streams (in millions)": 1820, "Energy": 0.66, "Danceability": 0.95},
    {"Artist Name": "Eminem", "Track Name": "Mockingbird", "Album": "Encore", "Release Year": 2004, "Streams (in millions)": 1710, "Energy": 0.89, "Danceability": 0.64},

    # BTS
    {"Artist Name": "BTS", "Track Name": "Dynamite", "Album": "BE", "Release Year": 2020, "Streams (in millions)": 1920, "Energy": 0.76, "Danceability": 0.75},
    {"Artist Name": "BTS", "Track Name": "Butter", "Album": "Butter", "Release Year": 2021, "Streams (in millions)": 1340, "Energy": 0.46, "Danceability": 0.76},
    {"Artist Name": "BTS", "Track Name": "Boy With Luv (feat. Halsey)", "Album": "Map of the Soul: Persona", "Release Year": 2019, "Streams (in millions)": 1180, "Energy": 0.86, "Danceability": 0.65},
    {"Artist Name": "BTS", "Track Name": "Life Goes On", "Album": "BE", "Release Year": 2020, "Streams (in millions)": 690, "Energy": 0.54, "Danceability": 0.57},
    {"Artist Name": "BTS", "Track Name": "Fake Love", "Album": "Love Yourself: Tear", "Release Year": 2018, "Streams (in millions)": 780, "Energy": 0.72, "Danceability": 0.52},

    # Kendrick Lamar
    {"Artist Name": "Kendrick Lamar", "Track Name": "HUMBLE.", "Album": "DAMN.", "Release Year": 2017, "Streams (in millions)": 2350, "Energy": 0.62, "Danceability": 0.90},
    {"Artist Name": "Kendrick Lamar", "Track Name": "All The Stars (with SZA)", "Album": "Black Panther Soundtrack", "Release Year": 2018, "Streams (in millions)": 1690, "Energy": 0.63, "Danceability": 0.70},
    {"Artist Name": "Kendrick Lamar", "Track Name": "Money Trees", "Album": "good kid, m.A.A.d city", "Release Year": 2012, "Streams (in millions)": 1820, "Energy": 0.53, "Danceability": 0.72},
    {"Artist Name": "Kendrick Lamar", "Track Name": "Not Like Us", "Album": "Single", "Release Year": 2024, "Streams (in millions)": 1100, "Energy": 0.47, "Danceability": 0.90},
    {"Artist Name": "Kendrick Lamar", "Track Name": "DNA.", "Album": "DAMN.", "Release Year": 2017, "Streams (in millions)": 1280, "Energy": 0.52, "Danceability": 0.64},

    # Bruno Mars
    {"Artist Name": "Bruno Mars", "Track Name": "Uptown Funk (with Mark Ronson)", "Album": "Uptown Special", "Release Year": 2014, "Streams (in millions)": 2150, "Energy": 0.61, "Danceability": 0.86},
    {"Artist Name": "Bruno Mars", "Track Name": "That's What I Like", "Album": "24K Magic", "Release Year": 2016, "Streams (in millions)": 2080, "Energy": 0.56, "Danceability": 0.85},
    {"Artist Name": "Bruno Mars", "Track Name": "Just the Way You Are", "Album": "Doo-Wops & Hooligans", "Release Year": 2010, "Streams (in millions)": 2150, "Energy": 0.84, "Danceability": 0.64},
    {"Artist Name": "Bruno Mars", "Track Name": "Locked Out of Heaven", "Album": "Unorthodox Jukebox", "Release Year": 2012, "Streams (in millions)": 2350, "Energy": 0.70, "Danceability": 0.73},
    {"Artist Name": "Bruno Mars", "Track Name": "Die With A Smile (with Lady Gaga)", "Album": "Single", "Release Year": 2024, "Streams (in millions)": 1250, "Energy": 0.68, "Danceability": 0.55},

    # Coldplay
    {"Artist Name": "Coldplay", "Track Name": "Yellow", "Album": "Parachutes", "Release Year": 2000, "Streams (in millions)": 2450, "Energy": 0.52, "Danceability": 0.43},
    {"Artist Name": "Coldplay", "Track Name": "Viva La Vida", "Album": "Viva la Vida or Death and All His Friends", "Release Year": 2008, "Streams (in millions)": 2210, "Energy": 0.62, "Danceability": 0.49},
    {"Artist Name": "Coldplay", "Track Name": "Something Just Like This (with The Chainsmokers)", "Album": "Memories...Do Not Open", "Release Year": 2017, "Streams (in millions)": 2680, "Energy": 0.64, "Danceability": 0.62},
    {"Artist Name": "Coldplay", "Track Name": "The Scientist", "Album": "A Rush of Blood to the Head", "Release Year": 2002, "Streams (in millions)": 2050, "Energy": 0.44, "Danceability": 0.56},
    {"Artist Name": "Coldplay", "Track Name": "Fix You", "Album": "X&Y", "Release Year": 2005, "Streams (in millions)": 1690, "Energy": 0.42, "Danceability": 0.21},

    # Olivia Rodrigo
    {"Artist Name": "Olivia Rodrigo", "Track Name": "drivers license", "Album": "SOUR", "Release Year": 2021, "Streams (in millions)": 2290, "Energy": 0.43, "Danceability": 0.59},
    {"Artist Name": "Olivia Rodrigo", "Track Name": "good 4 u", "Album": "SOUR", "Release Year": 2021, "Streams (in millions)": 2210, "Energy": 0.66, "Danceability": 0.56},
    {"Artist Name": "Olivia Rodrigo", "Track Name": "deja vu", "Album": "SOUR", "Release Year": 2021, "Streams (in millions)": 1580, "Energy": 0.61, "Danceability": 0.44},
    {"Artist Name": "Olivia Rodrigo", "Track Name": "traitor", "Album": "SOUR", "Release Year": 2021, "Streams (in millions)": 1520, "Energy": 0.34, "Danceability": 0.38},
    {"Artist Name": "Olivia Rodrigo", "Track Name": "vampire", "Album": "GUTS", "Release Year": 2023, "Streams (in millions)": 1150, "Energy": 0.53, "Danceability": 0.51},

    # Harry Styles
    {"Artist Name": "Harry Styles", "Track Name": "As It Was", "Album": "Harry's House", "Release Year": 2022, "Streams (in millions)": 3520, "Energy": 0.73, "Danceability": 0.52},
    {"Artist Name": "Harry Styles", "Track Name": "Watermelon Sugar", "Album": "Fine Line", "Release Year": 2019, "Streams (in millions)": 2720, "Energy": 0.82, "Danceability": 0.55},
    {"Artist Name": "Harry Styles", "Track Name": "Sign of the Times", "Album": "Harry Styles", "Release Year": 2017, "Streams (in millions)": 1650, "Energy": 0.57, "Danceability": 0.52},
    {"Artist Name": "Harry Styles", "Track Name": "Adore You", "Album": "Fine Line", "Release Year": 2019, "Streams (in millions)": 1720, "Energy": 0.77, "Danceability": 0.68},
    {"Artist Name": "Harry Styles", "Track Name": "Golden", "Album": "Fine Line", "Release Year": 2019, "Streams (in millions)": 1390, "Energy": 0.78, "Danceability": 0.45},

    # SZA
    {"Artist Name": "SZA", "Track Name": "Kill Bill", "Album": "SOS", "Release Year": 2022, "Streams (in millions)": 2150, "Energy": 0.73, "Danceability": 0.64},
    {"Artist Name": "SZA", "Track Name": "Snooze", "Album": "SOS", "Release Year": 2022, "Streams (in millions)": 1650, "Energy": 0.55, "Danceability": 0.56},
    {"Artist Name": "SZA", "Track Name": "Good Days", "Album": "SOS", "Release Year": 2020, "Streams (in millions)": 1280, "Energy": 0.66, "Danceability": 0.44},
    {"Artist Name": "SZA", "Track Name": "Nobody Gets Me", "Album": "SOS", "Release Year": 2022, "Streams (in millions)": 980, "Energy": 0.35, "Danceability": 0.42},
    {"Artist Name": "SZA", "Track Name": "The Weekend", "Album": "Ctrl", "Release Year": 2017, "Streams (in millions)": 1150, "Energy": 0.50, "Danceability": 0.65},

    # Arijit Singh
    {"Artist Name": "Arijit Singh", "Track Name": "Tum Hi Ho", "Album": "Aashiqui 2", "Release Year": 2013, "Streams (in millions)": 750, "Energy": 0.57, "Danceability": 0.45},
    {"Artist Name": "Arijit Singh", "Track Name": "Channa Mereya", "Album": "Ae Dil Hai Mushkil", "Release Year": 2016, "Streams (in millions)": 690, "Energy": 0.49, "Danceability": 0.48},
    {"Artist Name": "Arijit Singh", "Track Name": "Kesariya", "Album": "Brahmastra", "Release Year": 2022, "Streams (in millions)": 620, "Energy": 0.58, "Danceability": 0.57},
    {"Artist Name": "Arijit Singh", "Track Name": "Apna Bana Le", "Album": "Bhediya", "Release Year": 2022, "Streams (in millions)": 580, "Energy": 0.54, "Danceability": 0.51},
    {"Artist Name": "Arijit Singh", "Track Name": "Shayad", "Album": "Love Aaj Kal", "Release Year": 2020, "Streams (in millions)": 520, "Energy": 0.46, "Danceability": 0.47}
]

df_tracks = pd.DataFrame(data)
df_tracks.to_csv("E:/Sanskarjupyter/spotify_top_tracks.csv", index=False)
print("Saved spotify_top_tracks.csv with", len(df_tracks), "tracks.")
