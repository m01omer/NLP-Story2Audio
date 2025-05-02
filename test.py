# # import torch

# # # Check if CUDA is available
# # cuda_available = torch.cuda.is_available()

# # # Print whether CUDA is available
# # if cuda_available:
# #     print(f"CUDA is available. Version: {torch.version.cuda}")
# #     print(f"Current device: {torch.cuda.get_device_name(0)}")
# # else:
# #     print("CUDA is not available. Check your installation.")


# #-------------------------------

# import re

# class LanguageDetector:
#     """Detects if text is Roman Urdu or another language."""

#     def __init__(self):
#         self.roman_urdu_markers = set([
#             'hai', 'main', 'aur', 'kya', 'ko', 'se', 'par', 'ke', 'ka', 'ki',
#             'ap', 'tum', 'hum', 'mein', 'nahi', 'kuch', 'tha', 'ho', 'jata', 'karna',
#             'acha', 'theek', 'lekin', 'aese', 'kese', 'magar', 'phir', 'kyun', 'kahan',
#             'mujhe', 'tumhe', 'woh', 'kaun', 'kyun', 'kab', 'kaisa', 'sab', 'zindagi', 'pyar', 'dil',
#             'gya', 'raha', 'kr', 'hoga', 'tha', 'hoti', 'kisi', 'ab', 'chalo', 'bata', 'sun', 'acha'
#         ])
#         self.roman_urdu_patterns = ['ch', 'sh', 'kh', 'gh', 'ph', 'th', 'aa', 'ee', 'oo']
#         self.cache_size_limit = 500
#         self.detection_cache = {}
#         self.cache_hits = 0
#         self.cache_misses = 0

#     def detect_roman_urdu(self, text):
#         """Detect if text is likely Roman Urdu."""
#         cache_key = hash(text[:100])
#         if cache_key in self.detection_cache:
#             self.cache_hits += 1
#             return self.detection_cache[cache_key]
#         self.cache_misses += 1

#         if len(self.detection_cache) > self.cache_size_limit:
#             for k in list(self.detection_cache.keys())[:self.cache_size_limit // 2]:
#                 del self.detection_cache[k]

#         text_lower = text.lower()
#         words = re.findall(r'\b\w+\b', text_lower)

#         if not words:
#             self.detection_cache[cache_key] = False
#             return False

#         # Marker matches
#         marker_matches = sum(1 for word in words if word in self.roman_urdu_markers)
#         marker_ratio = marker_matches / len(words)

#         # Pattern frequency
#         pattern_matches = sum(len(re.findall(pat, text_lower)) for pat in self.roman_urdu_patterns)
#         pattern_score = min(pattern_matches / 10.0, 1.0)

#         # Heuristic: lots of vowels and specific consonants (a, k, n, r)
#         vowel_ratio = sum(text_lower.count(ch) for ch in 'aeiou') / max(len(text_lower), 1)
#         urdu_letter_score = sum(text_lower.count(ch) for ch in 'aknhr') / max(len(text_lower), 1)

#         # Final score
#         score = (marker_ratio * 0.5) + (pattern_score * 0.3) + (urdu_letter_score * 0.2)
#         is_roman_urdu = score > 0.3

#         self.detection_cache[cache_key] = is_roman_urdu
#         return is_roman_urdu

#     def get_cache_stats(self):
#         total = self.cache_hits + self.cache_misses
#         hit_rate = (self.cache_hits / total) * 100 if total > 0 else 0
#         return {
#             "cache_size": len(self.detection_cache),
#             "max_size": self.cache_size_limit,
#             "hits": self.cache_hits,
#             "misses": self.cache_misses,
#             "hit_rate": f"{hit_rate:.2f}%"
#         }


# ld = LanguageDetector()
# print(ld.detect_roman_urdu("Main aj school nahi jaa raha"))  # ✅ True
# print(ld.detect_roman_urdu("Today is a good day."))           # ❌ False
