# whiteboard-stitch

See http://joshuahhh.com/projects/whiteboard-stitch.

The source code to **whiteboard-stitch** is freely available for study. However, it is not packaged for turn-key operation, and you should not expect:
* the code to be easy to use, or
* anyone to assist you in use of the code.

I hope it is useful nonetheless.

## Dependencies

* opencv (with numpy)
* imutils
* numba
* clint
* yaml
* [possibly more?]

## Usage

To stitch together the concurrency whiteboard, run:

```
python stitch-cli.py whiteboards/concurrency/a.yaml
```

To generate image pyramids for the viewer, run:

```
python bundle.py whiteboards viewer/static/data
```

To build the viewer, go into the `viewer` directory and run:

```
yarn build
```
