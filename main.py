from anomaly_server import Server

import config

def main():

    s = Server(training=config.TRAIN)

    for user, passw in config.USERS.items():
        s.add_cam(user, passw)

    s.run()

if __name__ == '__main__':
    main()
