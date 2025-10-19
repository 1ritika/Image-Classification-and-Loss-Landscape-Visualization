
import sys

def usage():
    print("usage:")
    print("  python main.py cnn")
    print("  python main.py resnet <depth>")
    print("  python main.py plainnet <depth>")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)
    
    model_name = sys.argv[1].lower()

    if model_name == "cnn":
        from cnn import main as cnn_main
        cnn_main()
    elif model_name == "resnet":
        if len(sys.argv) < 3:
            print("for resnet, please provide overall depth (e.g. 20, 56, or 110)")
            usage()
            sys.exit(1)
        depth = int(sys.argv[2])
        from resnet import main as resnet_main
        resnet_main(depth)
    elif model_name == "plainnet":
        if len(sys.argv) < 3:
            print("for plainnet, please provide overall depth (e.g. 20, 56, or 110)")
            usage()
            sys.exit(1)
        depth = int(sys.argv[2])
        from plainnet import main as plainnet_main
        plainnet_main(depth)
    else:
        print(f"unknown model: {model_name}")
        usage()
