import fastf1

fastf1.Cache.enable_cache('cache')

season = 2026
round_number = 1  # Australian Grand Prix

session = fastf1.get_session(season, round_number, 'R')
session.load()

print(f"Loaded {session.event['EventName']} - {session.event['Country']}")

# Get lap data for all drivers
laps = session.laps

# Convert to DataFrame and save to CSV
laps.to_csv('data/f1_laps.csv', index=False)

print(f"Saved {len(laps)} laps to data/f1_laps.csv")