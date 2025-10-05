"""
CAG (Cache-Augmented Generation) Module for CivicMindAI
Handles intelligent caching of responses for performance optimization
"""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import os
from pathlib import Path

class CacheModule:
    """
    Cache Module for storing and retrieving frequently accessed civic information
    """
    
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.memory_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load persistent cache
        self._load_persistent_cache()
        
        logging.info(f"Cache module initialized with TTL: {ttl_hours} hours")
    
    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for a query
        """
        self.cache_stats["total_requests"] += 1
        
        try:
            cache_key = self._generate_cache_key(query)
            
            # Check memory cache first
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                if self._is_cache_valid(cached_item):
                    self.cache_stats["hits"] += 1
                    logging.debug(f"Memory cache hit for query: {query[:50]}")
                    return cached_item["response"]
                else:
                    # Remove expired item
                    del self.memory_cache[cache_key]
            
            # Check persistent cache
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_item = json.load(f)
                    
                    if self._is_cache_valid(cached_item):
                        self.cache_stats["hits"] += 1
                        # Load back to memory cache
                        self.memory_cache[cache_key] = cached_item
                        logging.debug(f"Persistent cache hit for query: {query[:50]}")
                        return cached_item["response"]
                    else:
                        # Remove expired file
                        cache_file.unlink()
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(f"Error reading cache file {cache_file}: {e}")
                    cache_file.unlink()
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logging.error(f"Error retrieving cached response: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    def cache_response(self, query: str, response: str, department: str, 
                      sources: List[str] = None, metadata: Dict[str, Any] = None):
        """
        Cache a response for future retrieval
        """
        try:
            cache_key = self._generate_cache_key(query)
            
            cached_item = {
                "query": query,
                "response": response,
                "department": department,
                "sources": sources or [],
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "cache_key": cache_key
            }
            
            # Store in memory cache
            self.memory_cache[cache_key] = cached_item
            
            # Store in persistent cache (async to avoid blocking)
            self._store_persistent_cache(cache_key, cached_item)
            
            logging.debug(f"Cached response for query: {query[:50]}")
            
        except Exception as e:
            logging.error(f"Error caching response: {e}")
    
    def _generate_cache_key(self, query: str) -> str:
        """
        Generate a unique cache key for the query
        """
        # Normalize query for better cache hits
        normalized_query = self._normalize_query(query)
        
        # Create hash of normalized query
        query_hash = hashlib.md5(normalized_query.encode('utf-8')).hexdigest()
        
        return f"civic_query_{query_hash}"
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query text for consistent caching
        """
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation that doesn't affect meaning
        chars_to_remove = ".,!?;:"
        for char in chars_to_remove:
            normalized = normalized.replace(char, "")
        
        # Standardize common area name variations
        area_variations = {
            "t nagar": "t. nagar",
            "tnagar": "t. nagar",
            "anna nagar": "anna nagar",
            "annanagar": "anna nagar",
            "adyar": "adyar",
            "velachery": "velachery"
        }
        
        for variation, standard in area_variations.items():
            normalized = normalized.replace(variation, standard)
        
        return normalized
    
    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """
        Check if cached item is still valid based on TTL
        """
        try:
            timestamp_str = cached_item.get("timestamp", "")
            cached_time = datetime.fromisoformat(timestamp_str)
            expiry_time = cached_time + timedelta(hours=self.ttl_hours)
            
            return datetime.now() < expiry_time
            
        except (ValueError, TypeError):
            return False
    
    def _store_persistent_cache(self, cache_key: str, cached_item: Dict[str, Any]):
        """
        Store cached item to persistent storage
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_item, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.warning(f"Error storing persistent cache: {e}")
    
    def _load_persistent_cache(self):
        """
        Load frequently accessed items from persistent cache to memory
        """
        try:
            if not self.cache_dir.exists():
                return
            
            # Load recent cache files to memory (last 100 files by modification time)
            cache_files = list(self.cache_dir.glob("*.json"))
            cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            loaded_count = 0
            for cache_file in cache_files[:100]:  # Load top 100 recent files
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_item = json.load(f)
                    
                    if self._is_cache_valid(cached_item):
                        cache_key = cache_file.stem
                        self.memory_cache[cache_key] = cached_item
                        loaded_count += 1
                    else:
                        # Remove expired file
                        cache_file.unlink()
                        
                except (json.JSONDecodeError, KeyError, OSError):
                    # Remove corrupted files
                    try:
                        cache_file.unlink()
                    except:
                        pass
            
            logging.info(f"Loaded {loaded_count} valid cache items from persistent storage")
            
        except Exception as e:
            logging.warning(f"Error loading persistent cache: {e}")
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """
        Clear cache items, optionally only those older than specified hours
        """
        try:
            items_cleared = 0
            
            if older_than_hours is None:
                # Clear all cache
                self.memory_cache.clear()
                
                # Clear persistent cache
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
                    items_cleared += 1
                    
                logging.info("Cleared all cache items")
                
            else:
                # Clear only old items
                cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
                
                # Clear from memory cache
                keys_to_remove = []
                for key, item in self.memory_cache.items():
                    try:
                        timestamp_str = item.get("timestamp", "")
                        cached_time = datetime.fromisoformat(timestamp_str)
                        if cached_time < cutoff_time:
                            keys_to_remove.append(key)
                    except (ValueError, TypeError):
                        keys_to_remove.append(key)  # Remove invalid timestamps
                
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    items_cleared += 1
                
                # Clear from persistent cache
                for cache_file in self.cache_dir.glob("*.json"):
                    try:
                        file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            cache_file.unlink()
                            items_cleared += 1
                    except OSError:
                        pass
                
                logging.info(f"Cleared {items_cleared} cache items older than {older_than_hours} hours")
                
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        """
        total_requests = self.cache_stats["total_requests"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate_percentage": round(hit_rate, 2),
            "memory_cache_size": len(self.memory_cache),
            "persistent_cache_files": len(list(self.cache_dir.glob("*.json"))),
            "cache_directory_size_mb": self._get_cache_dir_size()
        }
    
    def _get_cache_dir_size(self) -> float:
        """
        Get total size of cache directory in MB
        """
        try:
            total_size = 0
            for cache_file in self.cache_dir.glob("*.json"):
                total_size += cache_file.stat().st_size
            
            return round(total_size / (1024 * 1024), 2)  # Convert to MB
            
        except Exception:
            return 0.0
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular cached queries
        """
        try:
            # This is a simplified version - in production, you'd track query frequency
            popular_queries = []
            
            for cache_key, item in list(self.memory_cache.items())[:limit]:
                popular_queries.append({
                    "query": item.get("query", "")[:100],  # Truncate for display
                    "department": item.get("department", "Unknown"),
                    "cached_at": item.get("timestamp", ""),
                    "cache_key": cache_key
                })
            
            return popular_queries
            
        except Exception as e:
            logging.error(f"Error getting popular queries: {e}")
            return []
    
    def search_cache(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search cached responses for a term
        """
        try:
            search_results = []
            search_term_lower = search_term.lower()
            
            for cache_key, item in self.memory_cache.items():
                query = item.get("query", "").lower()
                response = item.get("response", "").lower()
                
                if search_term_lower in query or search_term_lower in response:
                    search_results.append({
                        "query": item.get("query", "")[:100],
                        "response": item.get("response", "")[:200],
                        "department": item.get("department", "Unknown"),
                        "relevance_score": self._calculate_relevance(search_term_lower, query, response)
                    })
            
            # Sort by relevance
            search_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return search_results[:10]  # Return top 10 results
            
        except Exception as e:
            logging.error(f"Error searching cache: {e}")
            return []
    
    def _calculate_relevance(self, search_term: str, query: str, response: str) -> float:
        """
        Calculate relevance score for search results
        """
        score = 0.0
        
        # Query matches are more important
        query_matches = query.count(search_term)
        score += query_matches * 2.0
        
        # Response matches
        response_matches = response.count(search_term)
        score += response_matches * 1.0
        
        return score
    
    def optimize_cache(self):
        """
        Optimize cache by removing least recently used items if cache is too large
        """
        try:
            max_memory_items = 1000  # Maximum items in memory cache
            max_persistent_files = 5000  # Maximum persistent cache files
            
            # Optimize memory cache
            if len(self.memory_cache) > max_memory_items:
                # Sort by timestamp and keep most recent
                sorted_items = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1].get("timestamp", ""),
                    reverse=True
                )
                
                # Keep most recent items
                self.memory_cache = dict(sorted_items[:max_memory_items])
                
                logging.info(f"Optimized memory cache to {len(self.memory_cache)} items")
            
            # Optimize persistent cache
            cache_files = list(self.cache_dir.glob("*.json"))
            if len(cache_files) > max_persistent_files:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                
                files_to_remove = cache_files[:-max_persistent_files]
                for cache_file in files_to_remove:
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass
                
                logging.info(f"Optimized persistent cache, removed {len(files_to_remove)} old files")
            
        except Exception as e:
            logging.error(f"Error optimizing cache: {e}")
    
    def export_cache_data(self) -> Dict[str, Any]:
        """
        Export cache data for analysis
        """
        try:
            return {
                "cache_stats": self.get_cache_stats(),
                "popular_queries": self.get_popular_queries(20),
                "memory_cache_sample": [
                    {
                        "query": item.get("query", "")[:100],
                        "department": item.get("department", ""),
                        "timestamp": item.get("timestamp", "")
                    }
                    for item in list(self.memory_cache.values())[:50]
                ],
                "export_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error exporting cache data: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """
        Cleanup when module is destroyed
        """
        try:
            # Ensure any pending cache writes are completed
            pass
        except:
            pass