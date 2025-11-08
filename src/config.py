class ConfigManager:
    @staticmethod
    def get(key, default=None):
        config = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
        }
        return config.get(key, default)
