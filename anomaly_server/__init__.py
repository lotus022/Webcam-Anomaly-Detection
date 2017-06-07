import logging; logging.basicConfig(level=logging.CRITICAL)

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

from keras.models import load_model

from anomaly_server import image_tools
from anomaly_server import log

from datetime import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

save_folder = None
anomoly_model = None
collect_images = True

last_files = {}

class Server(object):

    def __init__(self, host='0.0.0.0', port=21, home_path='ftp', training=False, model_path=os.path.join('models', 'anomoly_model.h5')):
        """
        Creates a server

        The Server class (this) is the main part of this application.
        It creates an instance of the FTP server and configures it to
        register anomalies with the model provided.

        The server has two modes: training and processing. Training
        will save every image into the training folder for manuel classification
        and processing will use a pretrained model to actual detect anomalies.

        Parameters
        ----------
        host : str
            Host address of the server
        port : int
            Server port of the server
        home_path : str
            The default save location of FTP files before processing
        training : bool
            True if the server is in training mode
        model_path : str
            Path to the anomaly model
        """
        self.host = host
        self.port = port
        self.home_path = home_path

        self.auth = DummyAuthorizer()

        global anomoly_model, save_folder, collect_images

        collect_images = training

        if collect_images:
            save_folder = "training"
        else:
            anomoly_model = load_model(model_path)
            save_folder = "images"

    def add_cam(self, username, passw):
        """
        Adds an FTP user to the server

        Parameters
        ----------
        username : str
            The username
        passw : str
            The password
        """
        self.auth.add_user(username, passw, self.home_path, perm="elradfmw")

    def run(self):
        """
        Runs the server.

        This method creates a special handler to process images as they
        are loaded into the FTP server and begins serving of the host:port
        configured in __init__.
        """
        class CamHandler(FTPHandler):

            def on_file_received(self, file_):

                date = datetime.now()

                path = os.path.join(save_folder, date.strftime("%m-%d-%Y"))
                os.makedirs(path, exist_ok=True)
                fn = os.path.join(path, date.isoformat().replace(":", ",") + ".jpg")
                os.rename(file_, fn)

                if self.username not in last_files:
                    last_files.update({self.username : None})

                print(fn, self.username)

                try:

                    if collect_images:
                        process_train(fn, self.username)
                    else:
                        process(fn, self.username)

                except Exception as e:
                    print(e)

            def on_incomplete_file_received(self, file_):
                os.remove(file_)

        handler = CamHandler
        handler.authorizer = self.auth

        server = FTPServer((self.host, self.port), handler)
        server.serve_forever()

def process_train(fn, username):
    """Handler for images during training mode."""
    last_file = last_files[username]

    if last_file:

        a, b = image_tools.load(last_file, fn)

        if image_tools.is_different(a, b):
            d_fn = fn.replace('.jpg', '.d.jpg')
            image_tools.create_delta_image(a, b, fn=d_fn, save=True)

    last_files[username] = fn

def process(fn, username):
    """Handler for images during processing mode."""
    last_file = last_files[username]

    if last_file:

        log.image_taken(fn, username)

        a, b = image_tools.load(last_file, fn)

        if image_tools.is_different(a, b):

            dimage = image_tools.create_delta_image(a, b, fn=fn.replace('.jpg','.d.jpg'), save=True)#
            anom = image_tools.is_anomoly(anomoly_model, dimage)

            if not anom:
                os.remove(last_file)
            else:
                log.anomoly(fn, username)

        else:
            os.remove(last_file)

    last_files[username] = fn
