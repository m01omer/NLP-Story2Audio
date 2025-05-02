
import re
import string

class LanguageDetector:
    """Detects if text is Roman Urdu or another language."""
    
    def __init__(self):
        # Common Roman Urdu words and patterns
        self.roman_urdu_markers = [
            'hai', 'main', 'aur', 'kya', 'ko', 'se', 'par', 'ke', 'ka', 'ki',
            'ap', 'tum', 'hum', 'mein', 'nahi', 'kuch', 'tha', 'ho', 'jata', 'karna',
            'acha', 'theek', 'lekin', 'aese', 'kese', 'magar', 'phir', 'kyun', 'kahan'
        ]
        self.roman_urdu_patterns = ['ch', 'sh', 'kh', 'gh', 'ph', 'th', 'aa', 'ee', 'oo']
        
        # Cache for detection results
        self.detection_cache = {}
        self.cache_size_limit = 500
        self.cache_hits = 0
        self.cache_misses = 0
    
    def detect_roman_urdu(self, text):
        """
        Detect if text is likely Roman Urdu based on patterns and character frequencies.
        Returns boolean indicating if text is likely Roman Urdu.
        """
        # Check cache first
        cache_key = hash(text[:100])  # Use first 100 chars to avoid long keys
        if cache_key in self.detection_cache:
            self.cache_hits += 1
            return self.detection_cache[cache_key]
        
        self.cache_misses += 1
        
        # Limit cache size
        if len(self.detection_cache) > self.cache_size_limit:
            # Simple strategy: clear half the cache when limit is reached
            keys = list(self.detection_cache.keys())
            for k in keys[:len(keys)//2]:
                del self.detection_cache[k]
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        words = text_lower.split()
        
        # No text to analyze
        if not words:
            return False
        
        # Check for Roman Urdu word patterns
        marker_count = sum(1 for word in words if word in self.roman_urdu_markers)
        
        # Check for specific character combinations common in Roman Urdu
        pattern_count = sum(text_lower.count(pattern) for pattern in self.roman_urdu_patterns)
        
        # Calculate probability score
        total_words = len(words)
        score = (marker_count / total_words) * 0.7 + min(pattern_count / 10, 1.0) * 0.3
        is_roman_urdu = score > 0.4  # Threshold can be adjusted
        
        # Cache the result
        self.detection_cache[cache_key] = is_roman_urdu
        
        return is_roman_urdu
    
    def get_cache_stats(self):
        """Return cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) * 100 if total > 0 else 0
        return {
            "cache_size": len(self.detection_cache),
            "max_size": self.cache_size_limit,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }
    


    #-----------------------------------------------------------------#

class TextPreprocessor:
    def __init__(self):
        # Common patterns for text normalization
        self.abbreviations = {
            "Mr.": "Mister",
            "Mrs.": "Misses",
            "Dr.": "Doctor",
            "Prof.": "Professor",
            "etc.": "etcetera",
            "e.g.": "for example",
            "i.e.": "that is",
            # Add more abbreviations as needed
        }
        
        # Cache for preprocessed text
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000  # Limit cache size
    
    def preprocess_text(self, text):
        """Apply preprocessing operations to optimize text for TTS inference"""
        # Check cache first
        if text in self.cache:
            self.cache_hits += 1
            return self.cache[text]
        
        self.cache_misses += 1
        
        # Start processing
        processed_text = text
        
        # Step 1: Normalize whitespace
        processed_text = re.sub(r'\s+', ' ', processed_text)
        processed_text = processed_text.strip()
        
        # Step 2: Expand abbreviations for better pronunciation
        for abbr, expansion in self.abbreviations.items():
            processed_text = processed_text.replace(abbr, expansion)
        
        # Step 3: Handle numbers - convert to spoken form
        # This is a simplified version - you might want more sophisticated logic
        processed_text = re.sub(r'(\d+)', r' \1 ', processed_text)
        processed_text = re.sub(r'\s+', ' ', processed_text)  # Clean up spaces again
        
        # Step 4: Handle special characters
        # Remove excessive punctuation
        processed_text = re.sub(r'([!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~])\1+', r'\1', processed_text)
        
        # Step 5: Add appropriate pauses (commas) for long sentences without punctuation
        # if len(processed_text) > 40 and ',' not in processed_text and '.' not in processed_text:
        #     words = processed_text.split()
        #     chunks = [' '.join(words[i:i+15]) for i in range(0, len(words), 15)]
        #     processed_text = ', '.join(chunks)
        
        # Store in cache
        if len(self.cache) >= self.max_cache_size:
            # Simple cache eviction - clear 25% of the oldest entries
            keys_to_remove = list(self.cache.keys())[:int(self.max_cache_size * 0.25)]
            for key in keys_to_remove:
                del self.cache[key]
        
        self.cache[text] = processed_text
        
        return processed_text
    def preprocess_roman_urdu(self,text): ## roman urdu ka pre processing
         
        if text in self.cache:
            self.cache_hits += 1
            return self.cache[text]
        
        self.cache_misses += 1
        
        # Start processing
        processed_text = text
        
        # Step 1: Basic normalization (similar to regular text)
        processed_text = re.sub(r'\s+', ' ', processed_text)
        processed_text = processed_text.strip()
        
        # Step 2: Handle repeated characters (common in Roman Urdu for emphasis)
        # Convert "zaraaa" to "zara" but preserve meaningful doubles like "zindagi"
        processed_text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', processed_text)
        
        # Step 3: Handle numeric substitutions commonly used in chat/SMS Roman Urdu
        numeric_substitutions = {
            r'\b2\b': 'to',        # Used as standalone "to" 
            r'\bm2\b': 'mein tu',  # "mein tu"
            r'\b4\b': 'for',       # English "for" in mixed text
            r'\bu\b': 'you',       # English "you" in mixed text
            r'\br\b': 'are',       # English "are" in mixed text
            r'(\w+)2(\w+)': r'\1 to \2',  # Used in between words
            r'(\w+)4(\w+)': r'\1 for \2', # Used in between words
            r'(\w+)8(\w+)': r'\1 ate \2', # Used in between words
            r'(\w*)9(\w*)': r'\1 nine \2' # Used in numbers
        }
        
        for pattern, replacement in numeric_substitutions.items():
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        # Step 4: Fix common Roman Urdu shorthand and spelling variations
        common_variations = {
            r'\btha\b': 'ta',             # Common pronunciation variation
            r'\bthay\b': 'tay',           # Common pronunciation variation
            r'\bhogya\b': 'ho gaya',      # Word separation
            r'\bhorha\b': 'ho raha',      # Word separation
            r'\bkrna\b': 'karna',         # Add vowels for better pronunciation
            r'\bkrenge\b': 'karenge',     # Add vowels
            r'\bkrein\b': 'karein',       # Add vowels
            r'\bkya\b': 'kia',            # Alternative spelling 
            r'\bnhi\b': 'nahi',           # Expand contracted forms
            r'\brhna\b': 'rehna',         # Add vowels
            r'\bsmjh\b': 'samajh',        # Add vowels
            r'\bh\b': 'hai',              # Common shorthand
            r'\bg\b': 'gaya',             # Common shorthand
            r'\bap\b': 'aap',             # More formal form
            r'\bmujy\b': 'mujhe',         # Spelling variation
            r'\btmhe\b': 'tumhe',         # Spelling variation
            r'\bahsta\b': 'aahista',      # Add proper vowels
            r'\bpochna\b': 'poochna',     # Double vowel for proper pronunciation
            r'\bpata\b': 'patta',         # Proper pronunciation
            r'\bht\b': 'bohat',           # Expand common abbreviation
            r'\bmgr\b': 'magar',          # Expand common abbreviation
            r'\blkn\b': 'lekin',          # Expand common abbreviation
            r'\bsth\b': 'saath',          # Expand common abbreviation
            r'\bky\b': 'kay',             # Expand common abbreviation
        }
        
        for pattern, replacement in common_variations.items():
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        # Step 5: Handle Urdu diacritics and pronunciation markers
        phonetic_markers = {
            r'ph': 'f',     # "ph" sound as in "phool" (flower)
            r'kh': 'x',     # Special marker for "kh" sound
            r'th': 'θ',     # Special marker for "th" sound
            r'ch': 'č',     # Special marker for "ch" sound
            r'sh': 'š',     # Special marker for "sh" sound
            r'zh': 'ž',     # Special marker for "zh" sound
        }
        
        # Only apply these replacements when they form specific sounds
        # We need to be careful not to break words where these are separate sounds
        for pattern, replacement in phonetic_markers.items():
            # Use lookbehind and lookahead to ensure we're not breaking words
            processed_text = re.sub(r'(?<!\w)' + pattern + r'(?!\w)', replacement, processed_text, flags=re.IGNORECASE)
            # Also replace at word boundaries
            processed_text = re.sub(r'\b' + pattern, replacement, processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(pattern + r'\b', replacement, processed_text, flags=re.IGNORECASE)
        
        # # Step 6: Add appropriate pauses for long sentences
        # if len(processed_text) > 80 and ',' not in processed_text and '.' not in processed_text:
        #     words = processed_text.split()
        #     # Shorter chunks for Roman Urdu - tends to have longer words
        #     chunks = [' '.join(words[i:i+10]) for i in range(0, len(words), 10)]
        #     processed_text = ', '.join(chunks)
        
        # Step 7: Fix English loanwords commonly used in Roman Urdu
        # We want to make sure these are pronounced correctly
        loanwords = {
            r'\bcomputer\b': 'kampyooter',
            r'\bmobile\b': 'mobail',
            r'\binternet\b': 'internat',
            r'\bcollege\b': 'kaalij',
            r'\buniversity\b': 'yooniversiti',
            r'\bschool\b': 'iskool',
            r'\bhospital\b': 'haspatal',
            r'\bdoctor\b': 'daktar',
            r'\bbus\b': 'bas',
            r'\btrain\b': 'tren',
            r'\bplane\b': 'plan',
        }
        
        for pattern, replacement in loanwords.items():
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        # Store in cache
        if len(self.cache) >= self.max_cache_size:
            keys_to_remove = list(self.cache.keys())[:int(self.max_cache_size * 0.25)]
            for key in keys_to_remove:
                del self.cache[key]
        
        self.cache[text] = processed_text
        
        return processed_text
    
    def preprocess_description(self, description):
        """Clean and promptify the voice description"""
        desc = ' '.join(description.strip().split())
        return f"Voice Agent: \"{desc}.\"" if not desc.endswith(('.', '!', '?')) else f"Voice Agent: \"{desc}\""
    
    def get_cache_stats(self):
        """Return cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) * 100 if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }
    


    #-----------------------------------------------------------------#

class RomanUrduTransliterator:
    """Class to handle transliteration from Roman Urdu to Urdu script."""
    
    def __init__(self):
        # Dictionary mapping Roman Urdu characters/combinations to Urdu Unicode characters
        self.mapping = {
            # Vowels
            'a': 'ا',
            'aa': 'آ',
            'i': 'ا',
            'ee': 'ی',
            'u': 'ا',
            'oo': 'و',
            'o': 'او',
            'e': 'ای',
            
            # Consonants
            'b': 'ب',
            'p': 'پ',
            't': 'ت',
            'tt': 'ٹ',
            'j': 'ج',
            'ch': 'چ',
            'h': 'ہ',
            'kh': 'خ',
            'd': 'د',
            'dd': 'ڈ',
            'z': 'ز',
            'r': 'ر',
            'rr': 'ڑ',
            's': 'س',
            'sh': 'ش',
            'ss': 'ص',
            'zz': 'ض',
            'ta': 'ط',
            'za': 'ظ',
            'ai': 'ع',
            'gh': 'غ',
            'f': 'ف',
            'q': 'ق',
            'k': 'ک',
            'g': 'گ',
            'l': 'ل',
            'm': 'م',
            'n': 'ن',
            'w': 'و',
            'v': 'و',
            'hh': 'ھ',
            'y': 'ی',
            
            # Common combinations
            'th': 'تھ',
            'dh': 'دھ',
            'ph': 'پھ',
            'bh': 'بھ',
            'kh': 'کھ',
            'gh': 'گھ',
            
            # Special characters
            '.': '۔',
            ',': '،',
            '?': '؟',
            
            # Common words/patterns - based on the LanguageDetector's roman_urdu_markers
            'allah': 'اللہ',
            'insha': 'انشا',
            'inshallah': 'انشااللہ',
            'mashallah': 'ماشااللہ',
            'jee': 'جی',
            'haan': 'ہاں',
            'nahi': 'نہیں',
            'aap': 'آپ',
            'ap': 'آپ',
            'main': 'میں',
            'mein': 'میں',
            'hai': 'ہے',
            'hum': 'ہم',
            'aur': 'اور',
            'se': 'سے',
            'par': 'پر',
            'ka': 'کا',
            'ki': 'کی',
            'ke': 'کے',
            'ko': 'کو',
            'kya': 'کیا',
            'tum': 'تم',
            'kuch': 'کچھ',
            'tha': 'تھا',
            'ho': 'ہو',
            'jata': 'جاتا',
            'karna': 'کرنا',
            'acha': 'اچھا',
            'theek': 'ٹھیک',
            'lekin': 'لیکن',
            'aese': 'ایسے',
            'kese': 'کیسے',
            'magar': 'مگر',
            'phir': 'پھر',
            'kyun': 'کیوں',
            'kahan': 'کہاں'
        }
        
        # Extended mapping for more accurate transliteration
        self.extended_mapping = {
            'tha': 'تھا',
            'thay': 'تھے',
            'raha': 'رہا',
            'rahay': 'رہے',
            'karo': 'کرو',
            'karein': 'کریں',
            'kyun': 'کیوں',
            'kaise': 'کیسے',
            'kya': 'کیا',
            'kahan': 'کہاں',
            'yahan': 'یہاں',
            'wahan': 'وہاں',
            'zindagi': 'زندگی',
            'mohabbat': 'محبت',
            'dost': 'دوست',
            'pyar': 'پیار',
            'ishq': 'عشق',
            'dil': 'دل'
        }
        
        # Cache for transliteration results
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000

    def transliterate(self, roman_text):
        """
        Main method to transliterate Roman Urdu to Urdu script.
        
        Args:
            roman_text (str): Text in Roman Urdu
            
        Returns:
            str: Transliterated text in Urdu script
        """
        # Check cache first
        if roman_text in self.cache:
            self.cache_hits += 1
            return self.cache[roman_text]
        
        self.cache_misses += 1
        
        # First pass: Basic transliteration
        urdu_text = self._transliterate_to_urdu(roman_text)
        
        # Second pass: Improve readability and fix common issues
        improved_text = self._improve_urdu_text(urdu_text)
        
        # Store in cache
        if len(self.cache) >= self.max_cache_size:
            # Simple cache eviction - clear 25% of the oldest entries
            keys_to_remove = list(self.cache.keys())[:int(self.max_cache_size * 0.25)]
            for key in keys_to_remove:
                del self.cache[key]
        
        self.cache[roman_text] = improved_text
        
        return improved_text
    
    def _transliterate_to_urdu(self, roman_text):
        """
        Core transliteration function from Roman Urdu to Urdu script.
        
        Args:
            roman_text (str): Text in Roman Urdu
            
        Returns:
            str: Raw transliterated text in Urdu script
        """
        # First, try to replace common words and phrases
        # This helps with special cases and improves accuracy
        for roman, urdu in {**self.mapping, **self.extended_mapping}.items():
            # Use word boundaries to avoid partial replacements
            # This regex looks for the roman pattern as a whole word
            pattern = r'\b' + roman + r'\b'
            roman_text = re.sub(pattern, urdu, roman_text, flags=re.IGNORECASE)
        
        # Process remaining text character by character with context awareness
        result = ''
        i = 0
        while i < len(roman_text):
            found = False
            
            # Try to match 3-character combinations first
            if i <= len(roman_text) - 3:
                three_chars = roman_text[i:i+3].lower()
                if three_chars in self.mapping:
                    result += self.mapping[three_chars]
                    i += 3
                    found = True
                    continue
            
            # Then try 2-character combinations
            if i <= len(roman_text) - 2:
                two_chars = roman_text[i:i+2].lower()
                if two_chars in self.mapping:
                    result += self.mapping[two_chars]
                    i += 2
                    found = True
                    continue
            
            # Finally, try single characters
            single_char = roman_text[i].lower()
            if single_char in self.mapping:
                result += self.mapping[single_char]
            else:
                # Keep original character if no mapping exists
                result += roman_text[i]
            
            i += 1
        
        # Apply post-processing rules for better readability
        result = result.replace('ا' + 'ا', 'ا')  # Remove duplicate alifs
        
        return result
    
    def _improve_urdu_text(self, urdu_text):
        """
        Apply improvements and corrections to the transliterated Urdu text.
        
        Args:
            urdu_text (str): Raw transliterated Urdu text
            
        Returns:
            str: Improved Urdu text
        """
        # Fix common issues
        corrections = {
            'کررہا': 'کر رہا',
            'ہورہا': 'ہو رہا',
            'جارہا': 'جا رہا',
            'آرہا': 'آ رہا',
        }
        
        for incorrect, correct in corrections.items():
            urdu_text = urdu_text.replace(incorrect, correct)
        
        # Add proper spacing between words if missing
        # This is a simplified approach and may need refinement
        urdu_text = re.sub(r'([ا-ے])([ب-ی])', r'\1 \2', urdu_text)
        
        # Ensure correct direction for Urdu text (right-to-left)
        urdu_text = '\u202B' + urdu_text + '\u202C'  # RTL embedding
        
        return urdu_text
    
    def get_cache_stats(self):
        """Return cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) * 100 if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }
    
    #-----------------------------------------------------------------#


    # class ModelManager:
    # def __init__(self):
    #     global DEVICE_MODE
    #     if DEVICE_MODE == 'cpu':
    #         self.device = "cpu"
    #     else:
    #         self.device = "cuda" if torch.cuda.is_available() else "cpu"
    #     logging.info(f"ModelManager initialized with device: {self.device}")
        
    #     # Initialize models dictionary
    #     self.models = {}
    #     self.tokenizers = {}
    #     self.processors = {}
    #     # Apply memory optimizations
    #     self.optimize_memory()
        
    #     # Load default model at startup
    #     self._load_default_model()
    
    # # def _load_default_model(self):
    # #     """Load the default English TTS model."""
    # #     logging.info("Loading default TTS model...")
    # #     model, prompt_tokenizer, description_tokenizer = load_model()
        
    # #     self.models["default"] = model
    # #     self.tokenizers["default"] = (prompt_tokenizer, description_tokenizer)
        
    # #     # Move model to device
    # #     self.models["default"].to(self.device)
    # #     # Enable gradient checkpointing to reduce memory usage
    # #     if hasattr(self.models["default"], "gradient_checkpointing_enable"):
    # #         self.models["default"].gradient_checkpointing_enable()
    # #     # Set model to evaluation mode
    # #     self.models["default"].eval()
    # #     logging.info("Default model loaded successfully")

    # ## used for t5 speech model 
    # def _load_default_model(self):
    #     """Load the default English TTS model."""
    #     logging.info("Loading default TTS model...")
    #     model, tokenizer, processor = load_model()  # Now captures the processor too
        
    #     self.models["default"] = model
    #     self.tokenizers["default"] = tokenizer
    #     self.processors["default"] = processor

    #     # Move model to device
    #     self.models["default"].to(self.device)
        
    #     # Enable gradient checkpointing if available
    #     if hasattr(self.models["default"], "gradient_checkpointing_enable"):
    #         self.models["default"].gradient_checkpointing_enable()
        
    #     # Set model to evaluation mode
    #     self.models["default"].eval()
    #     logging.info("Default model loaded successfully")


    # def _load_roman_urdu_model(self):
    #     """Load the Roman Urdu TTS model if not already loaded."""
    #     if "roman_urdu" not in self.models:
    #         logging.info("Loading Roman Urdu TTS model...")
    #         model, prompt_tokenizer, description_tokenizer = load_urdu_model()
            
    #         self.models["roman_urdu"] = model
    #         self.tokenizers["roman_urdu"] = (prompt_tokenizer, description_tokenizer)
            
    #         # Move model to device
    #         self.models["roman_urdu"].to(self.device)
    #         # Enable gradient checkpointing to reduce memory usage
    #         if hasattr(self.models["roman_urdu"], "gradient_checkpointing_enable"):
    #             self.models["roman_urdu"].gradient_checkpointing_enable()
    #         # Set model to evaluation mode
    #         self.models["roman_urdu"].eval()
    #         logging.info("Urdu model loaded successfully")
    # def optimize_memory(self):
    #     if torch.cuda.is_available():
    #         # Empty CUDA cache
    #         torch.cuda.empty_cache()
            
    #         # Print memory stats
    #         logging.info(f"CUDA Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated, "
    #                     f"{torch.cuda.memory_reserved()/1024**2:.2f}MB reserved")
            
    #         # Set memory fraction to use (adjust as needed)
    #         # This helps prevent OOM by limiting maximum memory usage
    #         try:
    #             import gc
    #             gc.collect()
    #             torch.cuda.memory.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
    #             logging.info("Set CUDA memory fraction to 0.8")
    #         except Exception as e:
    #             logging.warning(f"Could not set memory fraction: {e}")
    
    # # def get_model_for_language(self, language='English'):
    # #     """Get the appropriate model and tokenizers for the specified language.""" 
    # #     if language == "roman_urdu" and "roman_urdu" not in self.models: ## agar roman urdu model loaded  nahi hai to load karo
    # #         self._load_urdu_model()
        
    # #     # Return requested model or default if not available
    # #     if language in self.models:
    # #         return (
    # #             self.models[language],
    # #             self.tokenizers[language][0],  # prompt tokenizer
    # #             self.tokenizers[language][1]   # description tokenizer
    # #         )
    # #     else:
    # #         return (
    # #             self.models["default"],
    # #             self.tokenizers["default"][0],
    # #             self.tokenizers["default"][1]
    # #         )
    
    # def get_model_for_language(self, language='English'):
    #     """Get the appropriate model and tokenizers for the specified language.""" 
    #     if language == "roman_urdu" and "roman_urdu" not in self.models:
    #         self._load_roman_urdu_model()
        
    #     # Return requested model or default if not available
    #     if language in self.models:
    #         if language == "default":
    #             return (
    #                 self.models[language],
    #                 self.tokenizers[language],
    #                 self.processors[language]
    #             )
    #         else:
    #             return (
    #                 self.models[language],
    #                 self.tokenizers[language][0],  # prompt tokenizer
    #                 self.tokenizers[language][1]   # description tokenizer
    #             )
    #     else:
    #         return (
    #             self.models["default"],
    #             self.tokenizers["default"],
    #             self.processors["default"]
    #         )
    # def get_loaded_models(self): 
    #     """Return list of currently loaded models."""
    #     return list(self.models.keys())