"use strict";

/**
 * data.js
 *
 * 4-stage pipeline (simplified for browser):
 *  1. Read Raw Data from movies_metadata.csv
 *  2. Extract Features from overview (sub_genre & themes, via keyword rules)
 *  3. Consolidate Keywords into master lists (hard-coded rules)
 *  4. Finalize & Encode into one-hot feature vectors
 *
 * Exposed globals for script.js:
 *  - rawMovies: [{ id, title, overview }]
 *  - movieFeatures: [{ id, title, overview, subGenres, themes, featureVector }]
 *  - masterSubGenres: [string]
 *  - masterThemes: [string]
 */

let rawMovies = [];
let movieFeatures = [];
let masterSubGenres = [];
let masterThemes = [];

/* ---------- Canonical feature rules (simulate LLM + consolidation) ---------- */

const SUBGENRE_RULES = [
  {
    name: "Science Fiction",
    keywords: [
      "sci-fi",
      "science fiction",
      "spaceship",
      "space ",
      "spacecraft",
      "galaxy",
      "alien",
      "robot",
      "android",
      "futuristic",
      "future",
      "time travel",
      "outer space",
      "interstellar",
      "planet"
    ]
  },
  {
    name: "Fantasy",
    keywords: [
      "magic",
      "wizard",
      "witch",
      "dragon",
      "kingdom",
      "sorcerer",
      "mythical",
      "legend",
      "fairy tale",
      "elves",
      "dwarf"
    ]
  },
  {
    name: "Action Thriller",
    keywords: [
      "thriller",
      "chase",
      "assassin",
      "hitman",
      "spy",
      "espionage",
      "explosion",
      "shootout",
      "gunfight",
      "mercenary",
      "terrorist",
      "undercover"
    ]
  },
  {
    name: "Crime",
    keywords: [
      "crime",
      "criminal",
      "gangster",
      "mafia",
      "mob",
      "heist",
      "robbery",
      "bank robbery",
      "drug",
      "cartel",
      "detective",
      "police",
      "cop"
    ]
  },
  {
    name: "Horror",
    keywords: [
      "horror",
      "slasher",
      "killer",
      "serial killer",
      "ghost",
      "haunted",
      "possession",
      "demon",
      "zombie",
      "vampire",
      "monster",
      "creature",
      "curse"
    ]
  },
  {
    name: "Romantic Comedy",
    keywords: [
      "romantic comedy",
      "rom-com",
      "rom com",
      "romance",
      "dating",
      "wedding",
      "marriage",
      "love story",
      "fall in love",
      "relationship"
    ]
  },
  {
    name: "Drama",
    keywords: [
      "drama",
      "family drama",
      "character study",
      "emotional",
      "intense drama",
      "tragedy"
    ]
  },
  {
    name: "Animation",
    keywords: [
      "animated",
      "animation",
      "cartoon",
      "pixar",
      "disney"
    ]
  },
  {
    name: "War",
    keywords: [
      "war",
      "soldier",
      "army",
      "world war",
      "battlefield",
      "military",
      "combat"
    ]
  },
  {
    name: "Western",
    keywords: [
      "western",
      "cowboy",
      "sheriff",
      "frontier",
      "outlaw"
    ]
  },
  {
    name: "Mystery",
    keywords: [
      "mystery",
      "whodunit",
      "investigation",
      "detective",
      "sleuth",
      "solve the case"
    ]
  },
  {
    name: "Other",
    keywords: [] // fallback if nothing else matches
  }
];

const THEME_RULES = [
  {
    name: "Good vs Evil",
    keywords: [
      "good vs evil",
      "good and evil",
      "fight evil",
      "dark lord",
      "evil overlord",
      "hero battles evil"
    ]
  },
  {
    name: "Coming of Age",
    keywords: [
      "coming of age",
      "teenager",
      "high school",
      "growing up",
      "adolescence",
      "youth"
    ]
  },
  {
    name: "Redemption",
    keywords: [
      "redemption",
      "redeem himself",
      "redeem herself",
      "atonement",
      "atone",
      "seeking forgiveness",
      "forgiveness"
    ]
  },
  {
    name: "Sacrifice",
    keywords: [
      "sacrifice",
      "self-sacrifice",
      "gives up his life",
      "gives up her life",
      "lays down his life",
      "lays down her life"
    ]
  },
  {
    name: "Revenge",
    keywords: [
      "revenge",
      "vengeance",
      "seeks revenge",
      "retaliation",
      "avenges"
    ]
  },
  {
    name: "Love",
    keywords: [
      "love story",
      "in love",
      "true love",
      "romance",
      "romantic relationship"
    ]
  },
  {
    name: "Friendship",
    keywords: [
      "friendship",
      "best friends",
      "buddies",
      "companions",
      "bond"
    ]
  },
  {
    name: "Hero's Journey",
    keywords: [
      "hero's journey",
      "reluctant hero",
      "chosen one",
      "destiny",
      "epic quest"
    ]
  },
  {
    name: "Survival",
    keywords: [
      "survive",
      "survival",
      "stranded",
      "disaster",
      "apocalypse",
      "post-apocalyptic"
    ]
  },
  {
    name: "Crime & Justice",
    keywords: [
      "courtroom",
      "trial",
      "justice",
      "lawyer",
      "judge",
      "legal drama"
    ]
  }
];

function initMasterLists() {
  masterSubGenres = SUBGENRE_RULES.map((rule) => rule.name);
  masterThemes = THEME_RULES.map((rule) => rule.name);
}

/* ---------- Stage 1: read raw CSV into {id, title, overview} ---------- */

function parseMoviesCSV(csvText) {
  const lines = csvText.split(/\r?\n/);
  if (lines.length === 0) return [];

  const headerFields = parseCSVLine(lines[0]);
  const idIndex = headerFields.indexOf("id");
  const titleIndex = headerFields.indexOf("title");
  const overviewIndex = headerFields.indexOf("overview");

  if (idIndex === -1 || titleIndex === -1 || overviewIndex === -1) {
    throw new Error(
      "CSV is missing required columns: 'id', 'title', 'overview'."
    );
  }

  const movies = [];
  const MAX_MOVIES = 1000; // safety limit; adjust if you want more

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line || !line.trim()) continue;

    const cols = parseCSVLine(line);
    const maxIndex = Math.max(idIndex, titleIndex, overviewIndex);
    if (cols.length <= maxIndex) continue;

    const id = cols[idIndex];
    const title = cols[titleIndex];
    const overview = cols[overviewIndex];

    movies.push({ id, title, overview });

    if (movies.length >= MAX_MOVIES) break;
  }

  return movies;
}

/**
 * Parse a CSV line into fields.
 * Handles:
 *  - Commas inside double quotes
 *  - Escaped double quotes ("") inside quoted fields
 */
function parseCSVLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      // Escaped quote
      if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  result.push(current);
  return result;
}

/* ---------- Stages 2–4: extract + encode ---------- */

function extractFeaturesFromOverview(overview) {
  const text = overview.toLowerCase();
  const subGenresSet = new Set();
  const themesSet = new Set();

  SUBGENRE_RULES.forEach((rule) => {
    const hit = rule.keywords.some((kw) => text.includes(kw));
    if (hit) subGenresSet.add(rule.name);
  });

  THEME_RULES.forEach((rule) => {
    const hit = rule.keywords.some((kw) => text.includes(kw));
    if (hit) themesSet.add(rule.name);
  });

  // Fallback for sub-genres
  if (subGenresSet.size === 0) {
    subGenresSet.add("Other");
  }

  return {
    subGenres: Array.from(subGenresSet),
    themes: Array.from(themesSet)
  };
}

function encodeFeatures(subGenres, themes) {
  const vector = [];

  masterSubGenres.forEach((name) => {
    vector.push(subGenres.includes(name) ? 1 : 0);
  });

  masterThemes.forEach((name) => {
    vector.push(themes.includes(name) ? 1 : 0);
  });

  return vector;
}

/**
 * Main loader: run all 4 stages and populate global arrays.
 */
async function loadData() {
  initMasterLists();
  const resultElement = document.getElementById("result");

  try {
    if (resultElement) {
      resultElement.textContent = "Loading movies_metadata.csv...";
    }

    // IMPORTANT: CSV is assumed to be in the same folder as index.html
    const response = await fetch("movies_metadata.csv");
    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} while loading movies_metadata.csv`
      );
    }

    const csvText = await response.text();

    // Stage 1
    rawMovies = parseMoviesCSV(csvText);

    // Stages 2–4
    movieFeatures = rawMovies.map((movie) => {
      const features = extractFeaturesFromOverview(movie.overview);
      const featureVector = encodeFeatures(
        features.subGenres,
        features.themes
      );

      return {
        ...movie,
        subGenres: features.subGenres,
        themes: features.themes,
        featureVector
      };
    });

    if (resultElement) {
      resultElement.textContent = `Loaded ${movieFeatures.length} movies. Please select a movie and click "Get Recommendations".`;
    }
  } catch (error) {
    console.error("Error in loadData:", error);
    if (resultElement) {
      resultElement.textContent = "Error loading data: " + error.message;
    }
  }
}
