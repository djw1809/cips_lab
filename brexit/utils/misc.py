import webbrowser

def open_user(id):
    webbrowser.open('https://twitter.com/intent/user?user_id='+id)

def open_tweet(id):
    webbrowser.open('twitter.com/a/statuses/'+id)
