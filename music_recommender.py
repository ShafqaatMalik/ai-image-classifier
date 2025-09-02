"""
Personalized Music Recommender System using Spotify API
=====================================================

This module provides a comprehensive music recommendation system that uses:
- User listening history
- Song audio features  
- Artist popularity
- Collaborative and content-based filtering

Author: AI Assistant
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import json
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')


class SpotifyDataExtractor:
    """Handles Spotify API authentication and data extraction."""
    
    def __init__(self, client_id: str = None, client_secret: str = None, 
                 redirect_uri: str = None, username: str = None):
        """
        Initialize Spotify API client.
        
        Args:
            client_id: Spotify app client ID
            client_secret: Spotify app client secret
            redirect_uri: Redirect URI for authentication
            username: Spotify username
        """
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = redirect_uri or os.getenv('SPOTIFY_REDIRECT_URI')
        self.username = username or os.getenv('SPOTIFY_USERNAME')
        
        if not all([self.client_id, self.client_secret]):
            raise ValueError("Spotify credentials not found. Please set environment variables or pass them directly.")
        
        # Setup authentication
        self.scope = "user-library-read user-top-read playlist-read-private user-read-recently-played"
        
        # Try user authentication first, fall back to client credentials
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope,
                username=self.username
            ))
            self.auth_type = "user"
        except:
            # Fall back to client credentials (limited functionality)
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            ))
            self.auth_type = "client"
            print("Warning: Using client credentials. User-specific features will be limited.")
    
    def get_user_top_tracks(self, limit: int = 50, time_range: str = 'medium_term') -> List[Dict]:
        """
        Get user's top tracks.
        
        Args:
            limit: Number of tracks to retrieve (max 50)
            time_range: 'short_term', 'medium_term', or 'long_term'
        
        Returns:
            List of track dictionaries
        """
        if self.auth_type != "user":
            print("User authentication required for this feature.")
            return []
        
        try:
            results = self.sp.current_user_top_tracks(limit=limit, time_range=time_range)
            return results['items']
        except Exception as e:
            print(f"Error fetching top tracks: {e}")
            return []
    
    def get_user_saved_tracks(self, limit: int = 50) -> List[Dict]:
        """Get user's saved tracks."""
        if self.auth_type != "user":
            print("User authentication required for this feature.")
            return []
        
        try:
            tracks = []
            offset = 0
            while len(tracks) < limit:
                batch_size = min(50, limit - len(tracks))
                results = self.sp.current_user_saved_tracks(limit=batch_size, offset=offset)
                if not results['items']:
                    break
                tracks.extend([item['track'] for item in results['items']])
                offset += batch_size
            return tracks
        except Exception as e:
            print(f"Error fetching saved tracks: {e}")
            return []
    
    def get_recently_played(self, limit: int = 50) -> List[Dict]:
        """Get user's recently played tracks."""
        if self.auth_type != "user":
            print("User authentication required for this feature.")
            return []
        
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            return [item['track'] for item in results['items']]
        except Exception as e:
            print(f"Error fetching recently played tracks: {e}")
            return []
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """
        Get audio features for a list of tracks.
        
        Args:
            track_ids: List of Spotify track IDs
        
        Returns:
            List of audio feature dictionaries
        """
        try:
            # Spotify API allows max 100 tracks per request
            features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                batch_features = self.sp.audio_features(batch)
                features.extend([f for f in batch_features if f is not None])
            return features
        except Exception as e:
            print(f"Error fetching audio features: {e}")
            return []
    
    def get_artist_info(self, artist_ids: List[str]) -> List[Dict]:
        """Get artist information including popularity."""
        try:
            artists = []
            for i in range(0, len(artist_ids), 50):  # Max 50 artists per request
                batch = artist_ids[i:i+50]
                batch_artists = self.sp.artists(batch)
                artists.extend(batch_artists['artists'])
            return artists
        except Exception as e:
            print(f"Error fetching artist info: {e}")
            return []
    
    def search_tracks(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for tracks."""
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            return results['tracks']['items']
        except Exception as e:
            print(f"Error searching tracks: {e}")
            return []


class MusicRecommenderEngine:
    """Core recommendation engine using collaborative and content-based filtering."""
    
    def __init__(self, spotify_extractor: SpotifyDataExtractor):
        """
        Initialize the recommendation engine.
        
        Args:
            spotify_extractor: Configured SpotifyDataExtractor instance
        """
        self.spotify = spotify_extractor
        self.user_tracks_df = None
        self.audio_features_df = None
        self.artists_df = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        
    def load_user_data(self):
        """Load and process user's music data."""
        print("Loading user data...")
        
        # Get user tracks from various sources
        top_tracks = self.spotify.get_user_top_tracks(limit=50)
        saved_tracks = self.spotify.get_user_saved_tracks(limit=50)
        recent_tracks = self.spotify.get_recently_played(limit=50)
        
        # Combine all tracks
        all_tracks = top_tracks + saved_tracks + recent_tracks
        
        # Remove duplicates
        seen_ids = set()
        unique_tracks = []
        for track in all_tracks:
            if track['id'] not in seen_ids:
                unique_tracks.append(track)
                seen_ids.add(track['id'])
        
        if not unique_tracks:
            print("No user tracks found. Using sample data for demonstration.")
            # Search for some popular tracks as sample data
            sample_queries = ['pop music', 'rock hits', 'jazz classics', 'electronic music', 'hip hop']
            for query in sample_queries:
                tracks = self.spotify.search_tracks(query, limit=10)
                unique_tracks.extend(tracks)
        
        # Convert to DataFrame
        tracks_data = []
        for track in unique_tracks:
            tracks_data.append({
                'track_id': track['id'],
                'track_name': track['name'],
                'artist_id': track['artists'][0]['id'],
                'artist_name': track['artists'][0]['name'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms']
            })
        
        self.user_tracks_df = pd.DataFrame(tracks_data)
        print(f"Loaded {len(self.user_tracks_df)} tracks")
        
        # Get audio features
        self.load_audio_features()
        
        # Get artist information
        self.load_artist_data()
    
    def load_audio_features(self):
        """Load audio features for user tracks."""
        if self.user_tracks_df is None or self.user_tracks_df.empty:
            return
        
        print("Loading audio features...")
        track_ids = self.user_tracks_df['track_id'].tolist()
        features = self.spotify.get_audio_features(track_ids)
        
        if features:
            self.audio_features_df = pd.DataFrame(features)
            # Merge with track info
            self.audio_features_df = self.audio_features_df.merge(
                self.user_tracks_df, left_on='id', right_on='track_id', how='inner'
            )
            print(f"Loaded audio features for {len(self.audio_features_df)} tracks")
    
    def load_artist_data(self):
        """Load artist information."""
        if self.user_tracks_df is None or self.user_tracks_df.empty:
            return
        
        print("Loading artist data...")
        artist_ids = self.user_tracks_df['artist_id'].unique().tolist()
        artists = self.spotify.get_artist_info(artist_ids)
        
        if artists:
            artists_data = []
            for artist in artists:
                artists_data.append({
                    'artist_id': artist['id'],
                    'artist_name': artist['name'],
                    'artist_popularity': artist['popularity'],
                    'genres': artist['genres'],
                    'followers': artist['followers']['total']
                })
            
            self.artists_df = pd.DataFrame(artists_data)
            print(f"Loaded data for {len(self.artists_df)} artists")
    
    def prepare_features(self):
        """Prepare features for recommendation algorithms."""
        if self.audio_features_df is None or self.audio_features_df.empty:
            print("No audio features available")
            return None
        
        # Select numerical audio features
        feature_columns = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        # Add track and artist popularity
        feature_columns.extend(['popularity'])
        
        # Merge with artist data if available
        if self.artists_df is not None:
            features_df = self.audio_features_df.merge(
                self.artists_df, on='artist_id', how='left'
            )
            feature_columns.append('artist_popularity')
        else:
            features_df = self.audio_features_df.copy()
        
        # Prepare feature matrix
        feature_matrix = features_df[feature_columns].fillna(0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Apply PCA for dimensionality reduction
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(pca_features)
        features_df['cluster'] = clusters
        
        return features_df, pca_features
    
    def content_based_recommendations(self, num_recommendations: int = 10) -> List[Dict]:
        """
        Generate content-based recommendations using audio features.
        
        Args:
            num_recommendations: Number of recommendations to generate
        
        Returns:
            List of recommended tracks
        """
        features_df, pca_features = self.prepare_features()
        if features_df is None:
            return []
        
        # Calculate user profile (average of user's tracks features)
        user_profile = np.mean(pca_features, axis=0).reshape(1, -1)
        
        # Search for similar tracks
        search_queries = ['popular music', 'new releases', 'trending songs']
        candidate_tracks = []
        
        for query in search_queries:
            tracks = self.spotify.search_tracks(query, limit=50)
            candidate_tracks.extend(tracks)
        
        if not candidate_tracks:
            return []
        
        # Get audio features for candidate tracks
        candidate_ids = [track['id'] for track in candidate_tracks[:100]]  # Limit for API
        candidate_features = self.spotify.get_audio_features(candidate_ids)
        
        if not candidate_features:
            return []
        
        # Prepare candidate features
        candidate_df = pd.DataFrame(candidate_features)
        feature_columns = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        candidate_matrix = candidate_df[feature_columns].fillna(0)
        candidate_scaled = self.scaler.transform(candidate_matrix)
        candidate_pca = self.pca.transform(candidate_scaled)
        
        # Calculate similarities
        similarities = cosine_similarity(user_profile, candidate_pca)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:num_recommendations]
        
        recommendations = []
        for idx in top_indices:
            track_id = candidate_features[idx]['id']
            # Find original track info
            for track in candidate_tracks:
                if track['id'] == track_id:
                    recommendations.append({
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'],
                        'popularity': track['popularity'],
                        'similarity_score': similarities[idx],
                        'preview_url': track.get('preview_url'),
                        'external_url': track['external_urls']['spotify']
                    })
                    break
        
        return recommendations
    
    def cluster_based_recommendations(self, num_recommendations: int = 10) -> List[Dict]:
        """
        Generate recommendations based on user's music clusters.
        
        Args:
            num_recommendations: Number of recommendations to generate
        
        Returns:
            List of recommended tracks
        """
        features_df, pca_features = self.prepare_features()
        if features_df is None:
            return []
        
        # Find user's most common clusters
        cluster_counts = features_df['cluster'].value_counts()
        top_clusters = cluster_counts.head(3).index.tolist()
        
        recommendations = []
        
        # Search for tracks in similar clusters
        for cluster in top_clusters:
            cluster_tracks = features_df[features_df['cluster'] == cluster]
            if not cluster_tracks.empty:
                # Get representative features of this cluster
                cluster_center = np.mean(
                    pca_features[features_df['cluster'] == cluster], axis=0
                )
                
                # Search for new tracks similar to this cluster
                search_results = self.spotify.search_tracks(
                    f"genre:{cluster_tracks.iloc[0]['artist_name']}", limit=20
                )
                
                for track in search_results[:5]:  # Top 5 from each cluster
                    recommendations.append({
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'],
                        'popularity': track['popularity'],
                        'cluster': int(cluster),
                        'preview_url': track.get('preview_url'),
                        'external_url': track['external_urls']['spotify']
                    })
        
        return recommendations[:num_recommendations]
    
    def hybrid_recommendations(self, num_recommendations: int = 10) -> List[Dict]:
        """
        Generate hybrid recommendations combining multiple approaches.
        
        Args:
            num_recommendations: Number of recommendations to generate
        
        Returns:
            List of recommended tracks
        """
        # Get recommendations from different methods
        content_recs = self.content_based_recommendations(num_recommendations // 2)
        cluster_recs = self.cluster_based_recommendations(num_recommendations // 2)
        
        # Combine and deduplicate
        all_recs = content_recs + cluster_recs
        seen_ids = set()
        unique_recs = []
        
        for rec in all_recs:
            if rec['track_id'] not in seen_ids:
                unique_recs.append(rec)
                seen_ids.add(rec['track_id'])
                
                if len(unique_recs) >= num_recommendations:
                    break
        
        return unique_recs
    
    def get_user_music_profile(self) -> Dict:
        """
        Generate a comprehensive user music profile.
        
        Returns:
            Dictionary containing user's music preferences and statistics
        """
        if self.audio_features_df is None or self.audio_features_df.empty:
            return {}
        
        # Audio feature averages
        feature_columns = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        avg_features = self.audio_features_df[feature_columns].mean().to_dict()
        
        # Top artists and genres
        top_artists = self.user_tracks_df['artist_name'].value_counts().head(5).to_dict()
        
        if self.artists_df is not None:
            all_genres = []
            for genres in self.artists_df['genres']:
                all_genres.extend(genres)
            top_genres = pd.Series(all_genres).value_counts().head(5).to_dict()
        else:
            top_genres = {}
        
        # Popularity distribution
        popularity_stats = {
            'avg_track_popularity': self.user_tracks_df['popularity'].mean(),
            'avg_artist_popularity': self.artists_df['artist_popularity'].mean() if self.artists_df is not None else 0
        }
        
        return {
            'audio_features': avg_features,
            'top_artists': top_artists,
            'top_genres': top_genres,
            'popularity_stats': popularity_stats,
            'total_tracks': len(self.user_tracks_df)
        }


class MusicRecommenderSystem:
    """Main class for the music recommender system."""
    
    def __init__(self, client_id: str = None, client_secret: str = None,
                 redirect_uri: str = None, username: str = None):
        """
        Initialize the music recommender system.
        
        Args:
            client_id: Spotify app client ID
            client_secret: Spotify app client secret
            redirect_uri: Redirect URI for authentication
            username: Spotify username
        """
        self.spotify = SpotifyDataExtractor(client_id, client_secret, redirect_uri, username)
        self.engine = MusicRecommenderEngine(self.spotify)
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the system by loading user data."""
        print("Initializing Music Recommender System...")
        self.engine.load_user_data()
        self.is_initialized = True
        print("System initialized successfully!")
    
    def get_recommendations(self, method: str = 'hybrid', num_recommendations: int = 10) -> List[Dict]:
        """
        Get music recommendations.
        
        Args:
            method: 'content', 'cluster', or 'hybrid'
            num_recommendations: Number of recommendations
        
        Returns:
            List of recommended tracks
        """
        if not self.is_initialized:
            self.initialize()
        
        if method == 'content':
            return self.engine.content_based_recommendations(num_recommendations)
        elif method == 'cluster':
            return self.engine.cluster_based_recommendations(num_recommendations)
        elif method == 'hybrid':
            return self.engine.hybrid_recommendations(num_recommendations)
        else:
            raise ValueError("Method must be 'content', 'cluster', or 'hybrid'")
    
    def get_user_profile(self) -> Dict:
        """Get user's music profile."""
        if not self.is_initialized:
            self.initialize()
        
        return self.engine.get_user_music_profile()
    
    def print_recommendations(self, recommendations: List[Dict]):
        """Print recommendations in a formatted way."""
        if not recommendations:
            print("No recommendations found.")
            return
        
        print(f"\n🎵 Music Recommendations ({len(recommendations)} tracks)")
        print("=" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['track_name']} by {rec['artist_name']}")
            print(f"   Popularity: {rec['popularity']}/100")
            if 'similarity_score' in rec:
                print(f"   Similarity: {rec['similarity_score']:.3f}")
            if 'cluster' in rec:
                print(f"   Cluster: {rec['cluster']}")
            if rec.get('preview_url'):
                print(f"   Preview: {rec['preview_url']}")
            print(f"   Spotify: {rec['external_url']}")
            print()
    
    def print_user_profile(self, profile: Dict):
        """Print user profile in a formatted way."""
        if not profile:
            print("No user profile available.")
            return
        
        print("\n👤 Your Music Profile")
        print("=" * 30)
        
        # Audio features
        print("\n🎼 Audio Preferences:")
        features = profile.get('audio_features', {})
        for feature, value in features.items():
            print(f"  {feature.capitalize()}: {value:.3f}")
        
        # Top artists
        print("\n🎤 Top Artists:")
        for artist, count in profile.get('top_artists', {}).items():
            print(f"  {artist}: {count} tracks")
        
        # Top genres
        print("\n🎸 Top Genres:")
        for genre, count in profile.get('top_genres', {}).items():
            print(f"  {genre}: {count} tracks")
        
        # Statistics
        print("\n📊 Statistics:")
        stats = profile.get('popularity_stats', {})
        print(f"  Total tracks analyzed: {profile.get('total_tracks', 0)}")
        print(f"  Average track popularity: {stats.get('avg_track_popularity', 0):.1f}/100")
        print(f"  Average artist popularity: {stats.get('avg_artist_popularity', 0):.1f}/100")


def main():
    """Demonstration of the music recommender system."""
    print("🎵 Personalized Music Recommender System")
    print("=" * 40)
    
    # Check for credentials
    if not all([os.getenv('SPOTIFY_CLIENT_ID'), os.getenv('SPOTIFY_CLIENT_SECRET')]):
        print("⚠️  Spotify API credentials not found!")
        print("Please set up your environment variables:")
        print("1. Create a Spotify app at https://developer.spotify.com/dashboard/applications")
        print("2. Copy .env.example to .env and fill in your credentials")
        print("3. Run the system again")
        return
    
    try:
        # Initialize the recommender system
        recommender = MusicRecommenderSystem()
        
        # Initialize with user data
        recommender.initialize()
        
        # Get and display user profile
        profile = recommender.get_user_profile()
        recommender.print_user_profile(profile)
        
        # Get different types of recommendations
        print("\n" + "="*60)
        
        # Content-based recommendations
        print("\n🔍 Content-Based Recommendations:")
        content_recs = recommender.get_recommendations('content', 5)
        recommender.print_recommendations(content_recs)
        
        # Cluster-based recommendations
        print("\n🎯 Cluster-Based Recommendations:")
        cluster_recs = recommender.get_recommendations('cluster', 5)
        recommender.print_recommendations(cluster_recs)
        
        # Hybrid recommendations
        print("\n🚀 Hybrid Recommendations (Best of Both):")
        hybrid_recs = recommender.get_recommendations('hybrid', 10)
        recommender.print_recommendations(hybrid_recs)
        
        print("\n✅ Recommendation system completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your Spotify API credentials")
        print("2. Make sure you have an active internet connection")
        print("3. Verify your Spotify app settings")


if __name__ == "__main__":
    main()