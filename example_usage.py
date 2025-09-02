"""
Example Usage of the Music Recommender System
===========================================

This script demonstrates how to use the music recommender system
with different configuration options and use cases.
"""

from music_recommender import MusicRecommenderSystem, SpotifyDataExtractor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_basic_usage():
    """Basic usage example with environment variables."""
    print("🎵 Basic Usage Example")
    print("-" * 30)
    
    # Initialize with environment variables
    recommender = MusicRecommenderSystem()
    
    # Get recommendations
    recommendations = recommender.get_recommendations(method='hybrid', num_recommendations=5)
    
    # Display results
    recommender.print_recommendations(recommendations)
    
    # Get user profile
    profile = recommender.get_user_profile()
    recommender.print_user_profile(profile)


def example_custom_credentials():
    """Example with custom credentials."""
    print("🔐 Custom Credentials Example")
    print("-" * 30)
    
    # You can pass credentials directly
    recommender = MusicRecommenderSystem(
        client_id="your_client_id",
        client_secret="your_client_secret", 
        redirect_uri="http://localhost:8080/callback",
        username="your_username"
    )
    
    # Rest of the usage is the same
    recommendations = recommender.get_recommendations('content', 3)
    recommender.print_recommendations(recommendations)


def example_different_methods():
    """Example showing different recommendation methods."""
    print("🎯 Different Methods Example")
    print("-" * 30)
    
    recommender = MusicRecommenderSystem()
    
    # Content-based recommendations
    print("\n1. Content-Based Recommendations:")
    content_recs = recommender.get_recommendations('content', 3)
    recommender.print_recommendations(content_recs)
    
    # Cluster-based recommendations  
    print("\n2. Cluster-Based Recommendations:")
    cluster_recs = recommender.get_recommendations('cluster', 3)
    recommender.print_recommendations(cluster_recs)
    
    # Hybrid recommendations
    print("\n3. Hybrid Recommendations:")
    hybrid_recs = recommender.get_recommendations('hybrid', 5)
    recommender.print_recommendations(hybrid_recs)


def example_advanced_usage():
    """Advanced usage with direct access to components."""
    print("🔬 Advanced Usage Example")
    print("-" * 30)
    
    # Initialize components separately for more control
    spotify = SpotifyDataExtractor()
    
    # Get user's top tracks
    top_tracks = spotify.get_user_top_tracks(limit=10, time_range='short_term')
    print(f"Found {len(top_tracks)} top tracks from the last 4 weeks")
    
    # Get recently played
    recent = spotify.get_recently_played(limit=5)
    print(f"Found {len(recent)} recently played tracks")
    
    # Get audio features for specific tracks
    if top_tracks:
        track_ids = [track['id'] for track in top_tracks[:3]]
        features = spotify.get_audio_features(track_ids)
        print(f"Got audio features for {len(features)} tracks")
        
        # Display some features
        for i, feature in enumerate(features):
            if feature:
                print(f"Track {i+1}: Danceability={feature['danceability']:.3f}, "
                      f"Energy={feature['energy']:.3f}, Valence={feature['valence']:.3f}")


def example_error_handling():
    """Example with proper error handling."""
    print("⚠️  Error Handling Example")
    print("-" * 30)
    
    try:
        # This will work if credentials are set
        recommender = MusicRecommenderSystem()
        recommendations = recommender.get_recommendations('hybrid', 5)
        
        if recommendations:
            print("✅ Successfully got recommendations!")
            recommender.print_recommendations(recommendations[:2])  # Show first 2
        else:
            print("⚠️  No recommendations found - this might be normal for new users")
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Make sure to set up your Spotify API credentials")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("💡 Check your internet connection and API credentials")


def demo_without_user_auth():
    """Demo using only client credentials (limited functionality)."""
    print("🔧 Client Credentials Demo")
    print("-" * 30)
    
    try:
        # Create extractor with client credentials only
        spotify = SpotifyDataExtractor()
        
        # Search for some popular tracks
        tracks = spotify.search_tracks("popular music 2024", limit=5)
        
        if tracks:
            print("Found popular tracks:")
            for i, track in enumerate(tracks, 1):
                print(f"{i}. {track['name']} by {track['artists'][0]['name']}")
                
            # Get audio features
            track_ids = [track['id'] for track in tracks]
            features = spotify.get_audio_features(track_ids)
            
            print(f"\nGot audio features for {len(features)} tracks")
        
    except Exception as e:
        print(f"Error in demo: {e}")


def main():
    """Run all examples."""
    print("🎵 Music Recommender System - Usage Examples")
    print("=" * 50)
    
    # Check if credentials are available
    if not all([os.getenv('SPOTIFY_CLIENT_ID'), os.getenv('SPOTIFY_CLIENT_SECRET')]):
        print("⚠️  No Spotify credentials found!")
        print("Please set up your .env file with:")
        print("- SPOTIFY_CLIENT_ID")
        print("- SPOTIFY_CLIENT_SECRET") 
        print("- SPOTIFY_REDIRECT_URI")
        print("- SPOTIFY_USERNAME")
        print("\nRunning limited demo...")
        demo_without_user_auth()
        return
    
    # Run examples
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Different Methods", example_different_methods),
        ("Advanced Usage", example_advanced_usage),
        ("Error Handling", example_error_handling)
    ]
    
    for name, func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            func()
        except Exception as e:
            print(f"Error in {name}: {e}")
        print()


if __name__ == "__main__":
    main()