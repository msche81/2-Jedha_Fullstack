def main():
    print('Cakes are the best!')
    with open("favorites_cakes.txt") as f:
        print(f.read())

if __name__ == "__main__":
    main()