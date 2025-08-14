"""
Configuration file for Amharic LLM Data Collection
"""

# Data sources configuration
DATA_SOURCES = {
    "huggingface": {
        "amharic_news_classification": {
            "dataset": "rasyosef/amharic-news-category-classification",
            "task": "classification",
            "max_samples": 1000
        },
        "amharic_llama_mt": {
            "dataset": "EthioNLP/Amharic_LLAMA_MT",
            "task": "translation",
            "max_samples": 1000
        },
        "afridocmt": {
            "dataset": "masakhane/AfriDocMT",
            "subset": "doc_health",
            "task": "translation",
            "max_samples": 500
        },
        "amharic_news_israel": {
            "dataset": "israel/Amharic-News-Text-classification-Dataset",
            "task": "classification",
            "max_samples": 1000
        }
    },
    
    "web_scraping": {
        "bbc_amharic": {
            "url": "https://www.bbc.com/amharic",
            "task": "news",
            "max_pages": 100
        },
        "voa_amharic": {
            "url": "https://amharic.voanews.com",
            "task": "news",
            "max_pages": 100
        },
        "dw_amharic": {
            "url": "https://www.dw.com/am",
            "task": "news", 
            "max_pages": 100
        },
        "wikimezmur": {
            "url": "https://wikimezmur.org/am",
            "task": "lyrics",
            "max_pages": 200
        },
        "ethiopian_folktales": {
            "url": "https://www.ethiopianfolktales.com/am",
            "task": "stories",
            "max_pages": 50
        }
    }
}

# Instruction templates for each task
INSTRUCTION_TEMPLATES = {
    "sentiment": [
        "የሚከተለውን ጽሑፍ ስሜት ተንትን፡ {text}",
        "ይህ ጽሑፍ አዎንታዊ፣ አሉታዊ ወይስ ገለልተኛ ነው? {text}",
        "የዚህን ጽሑፍ ስሜት ግለጽ፡ {text}",
        "ከታች ያለው ጽሑፍ ምን ዓይነት ስሜት ያሳያል? {text}",
        "ስሜቱን ለይ፡ {text}",
        "ይህ መልእክት ደስተኛ፣ ያዘነ ወይስ ገለልተኛ ነው? {text}",
        "የጽሑፉን ስሜታዊ ሁኔታ አመልክት፡ {text}"
    ],
    
    "classification": [
        "የሚከተለውን ዜና ወደ ምድብ አስገባ፡ {text}",
        "ይህ ዜና የትኛው ምድብ ነው? {text}",
        "ዜናውን መድብ፡ {text}",
        "የዚህ ጽሑፍ ዓይነት ምንድን ነው? {text}",
        "ወደ ትክክለኛው ምድብ አስገባ፡ {text}"
    ],
    
    "ner": [
        "ከዚህ ጽሑፍ ውስጥ የሰዎችን ስም ዘርዝር፡ {text}",
        "በዚህ ውስጥ የተጠቀሱ ሰዎች እነማን ናቸው? {text}",
        "ስሞችን ብቻ አውጣ፡ {text}",
        "የሰው ስሞችን ለይ፡ {text}",
        "በጽሑፉ ውስጥ ያሉ ሰዎችን ስም ጻፍ፡ {text}"
    ],
    
    "summarization": [
        "የሚከተለውን ጽሑፍ አጠቃልል፡ {text}",
        "አጭር ማጠቃለያ ስጥ፡ {text}",
        "በአጭሩ ግለጽ፡ {text}",
        "ዋናውን ሃሳብ ብቻ ጻፍ፡ {text}",
        "በጥቂት ዓረፍተ ነገሮች አጠቃልል፡ {text}"
    ],
    
    "translation": [
        "ወደ አማርኛ ተርጉም፡ {text}",
        "ወደ እንግሊዝኛ ተርጉም፡ {text}",
        "ትርጉሙን ስጥ፡ {text}",
        "በአማርኛ እንዴት ይባላል? {text}",
        "በእንግሊዝኛ እንዴት ይባላል? {text}"
    ],
    
    "qa": [
        "የሚከተለውን ጥያቄ መልስ፡ {question}\nዐውድ፡ {context}",
        "ከዐውዱ በመነሳት መልስ ስጥ፡\nጥያቄ፡ {question}\nዐውድ፡ {context}",
        "ትክክለኛውን መልስ ፈልግ፡\n{question}\n{context}",
        "በዐውዱ መሰረት ጥያቄውን መልስ፡ {question}\n{context}"
    ],
    
    "generation": [
        "ስለ {topic} አጭር ታሪክ ጻፍ",
        "የ{topic}ን ጽሑፍ ቀጥል፡ {text}",
        "{topic} የሚል ግጥም ጻፍ",
        "ስለ {topic} አብራራ",
        "{topic}ን የሚገልጽ ጽሑፍ ፍጠር"
    ]
}

# Quality filter thresholds
QUALITY_FILTERS = {
    "min_length": 10,  # Minimum text length in characters
    "max_length": 5000,  # Maximum text length
    "min_amharic_ratio": 0.7,  # Minimum ratio of Amharic characters
    "max_repetition_ratio": 0.3,  # Maximum repetition allowed
    "perplexity_threshold": 1000,  # Maximum perplexity score
    "toxicity_threshold": 0.8  # Maximum toxicity score
}

# API configurations (add your keys to .env file)
API_CONFIG = {
    "openai": {
        "model": "gpt-4-turbo-preview",
        "max_tokens": 2000,
        "temperature": 0.7
    },
    "anthropic": {
        "model": "claude-3-opus-20240229",
        "max_tokens": 2000,
        "temperature": 0.7
    }
}

# Output paths
OUTPUT_PATHS = {
    "raw_data": "data/raw",
    "processed_data": "data/processed",
    "synthetic_data": "data/synthetic",
    "final_dataset": "data/final_amharic_dataset.jsonl",
    "statistics": "data/dataset_statistics.json"
}

# Training configurations
TRAINING_CONFIG = {
    "train_split": 0.9,
    "val_split": 0.05,
    "test_split": 0.05,
    "seed": 42,
    "shuffle": True
}
