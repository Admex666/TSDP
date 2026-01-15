/**
 * API Client for Player Tracker
 */

const API_BASE_URL = 'http://localhost:8003/api/v1';

class PlayerTrackerAPI {
    /**
     * Get list of players with optional filters
     */
    async getPlayers(filters = {}) {
        const params = new URLSearchParams();
        
        if (filters.tracked !== undefined) params.append('tracked', filters.tracked);
        if (filters.position) params.append('position', filters.position);
        if (filters.league_id) params.append('league_id', filters.league_id);
        if (filters.min_minutes) params.append('min_minutes', filters.min_minutes);
        
        const url = `${API_BASE_URL}/players?${params}`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`Failed to fetch players: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Get detailed player information
     */
    async getPlayer(playerId) {
        const response = await fetch(`${API_BASE_URL}/players/${playerId}`);
        
        if (!response.ok) {
            throw new Error(`Failed to fetch player: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Get player statistics
     */
    async getPlayerStats(playerId, filters = {}) {
        const params = new URLSearchParams();
        
        if (filters.start_date) params.append('start_date', filters.start_date);
        if (filters.end_date) params.append('end_date', filters.end_date);
        if (filters.per90) params.append('per90', filters.per90);
        
        const url = `${API_BASE_URL}/players/${playerId}/stats?${params}`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`Failed to fetch player stats: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Update player information
     */
    async updatePlayer(playerId, data) {
        const response = await fetch(`${API_BASE_URL}/players/${playerId}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`Failed to update player: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Get list of leagues
     */
    async getLeagues() {
        const response = await fetch(`${API_BASE_URL}/leagues`);
        
        if (!response.ok) {
            throw new Error(`Failed to fetch leagues: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    /**
     * Import player stats from CSV
     */
    async importStats(file, source, autoMatch = true) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('source', source);
        formData.append('auto_match_players', autoMatch);
        
        const response = await fetch(`${API_BASE_URL}/import/stats`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Failed to import stats: ${response.statusText}`);
        }
        
        return await response.json();
    }
}

// Create global API instance
const api = new PlayerTrackerAPI();
