from datetime import datetime
import os

def _write(items):

    date = datetime.today()

    with open(os.path.join("images",  date.strftime("%m-%d-%Y")  + ".log"), 'a') as f:
        f.write("#".join(items) + '\n')

def _fn_to_date(fn):

    path, file_ = os.path.split(fn)

    return file_.replace('.jpg','').replace(',',':')

def image_taken(fn, username):
    """Log the capture of an image."""
    _write(['CAP', username, _fn_to_date(fn)])

def anomoly(fn, username):
    """Log an amomaly found with the given image."""
    _write(['ANO', username, _fn_to_date(fn)])
