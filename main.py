from anomaly_server import Server

import config

def main():

    s = Server(training=config.TRAIN) # Create Server

    for user, passw in config.USERS.items(): # Load Users
        s.add_cam(user, passw)

    s.run() # run.forever()

if __name__ == '__main__':
    main()
