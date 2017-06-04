
def main():

    s = Server(training=False)
    s.add_cam('yuncam', 'aBc1to3')
    s.run()

if __name__ == '__main__':
    main()
