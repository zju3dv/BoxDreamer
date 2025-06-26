# ask user to get proxy server and test it
import os
import sys
import argparse
import subprocess
import time


def start(proxy: str) -> None:
    # set proxy to environment and test it
    if proxy == '':
        proxy = input("Please input the proxy server: ")
    else:
        print(f"Proxy server is {proxy}.")
        
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    os.environ["all_proxy"] = proxy.replace("http://", "socks5://")
    
    # check proxy server
    print("Testing proxy server...")
    try:
        subprocess.run(["curl", "www.google.com"], check=True)
        print("Proxy server is working.")
    except subprocess.CalledProcessError:
        print("Proxy server is not working.")
        stop()
        sys.exit(1)

def stop():
    # unset proxy
    os.environ.pop("HTTP_PROXY")
    os.environ.pop("HTTPS_PROXY")
    print("Proxy server is stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Proxy server helper.')
    parser.add_argument('action', type=str, help='action name (start, stop)')
    parser.add_argument('--proxy', type=str, help='proxy server', default='')
    args = parser.parse_args()
    args.proxy
    # call corresponding function
    function_name = f"{args.action}"
    if function_name  == 'start':
        start(args.proxy)