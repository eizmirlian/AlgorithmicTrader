## PUT YOUR OPENAI API KET HERE ###
OPENAI_API_KEY = ''

#If you have a paid subscription with OpenAI, set this to false
FREE_VERSION = True

def get_config_params():
    if OPENAI_API_KEY == '':
        return 'OpenAI API Key not set'
    else:
        return OPENAI_API_KEY, FREE_VERSION