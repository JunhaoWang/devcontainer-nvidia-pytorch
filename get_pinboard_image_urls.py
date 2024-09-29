kwargs = {'arg_path': 'sandramedinadom/city-clothes', 'arg_dir': 'images', 'arg_thread_max': 0, 
 'arg_cut': -1, 'arg_board_timestamp': False, 'arg_log_timestamp': False, 
 'arg_force': False, 'arg_exclude_section': False, 'arg_rescrape': False, 
 'arg_img_only': False, 'arg_v_only': False, 'arg_update_all': False, 
 'arg_https_proxy': None, 'arg_http_proxy': None, 'arg_cookies': None}


import importlib.util
import sys

# Define the path to the pinterest-downloader.py file
file_path = './pinterest-downloader/pinterest-downloader.py'

# Load the module
spec = importlib.util.spec_from_file_location("pinterest_downloader_module", file_path)
pinterest_downloader_module = importlib.util.module_from_spec(spec)
sys.modules["pinterest_downloader_module"] = pinterest_downloader_module
spec.loader.exec_module(pinterest_downloader_module)

# Now you can access the function run_library_main
image_urls = pinterest_downloader_module.run_library_main(**kwargs)

print(image_urls)
