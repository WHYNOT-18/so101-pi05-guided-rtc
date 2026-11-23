"""Explicit RGB image transport. PNG is lossless; JPEG95 is opt-in lossy."""
import io
import numpy as np
from PIL import Image

SUPPORTED_IMAGE_CODECS=('raw','png','jpeg95')


def encode_observation(observation, codec):
    if codec not in SUPPORTED_IMAGE_CODECS:
        raise ValueError('Unsupported image codec')
    if codec=='raw': return observation
    output=dict(observation)
    for name in ('front','wrist'):
        value=np.asarray(observation[name])
        if value.shape!=(224,224,3) or value.dtype!=np.uint8:
            raise ValueError('Expected uint8 RGB224 image')
        stream=io.BytesIO()
        options=dict(quality=95,subsampling=0) if codec=='jpeg95' else dict(compress_level=3)
        Image.fromarray(value).save(stream,format='JPEG' if codec=='jpeg95' else 'PNG',**options)
        output[name]=stream.getvalue()
    output['_image_transport']=dict(version=1,codec=codec,color='RGB',shape=[224,224,3])
    return output


def decode_observation(observation):
    descriptor=observation.get('_image_transport')
    if descriptor is None: return observation
    if not isinstance(descriptor,dict) or descriptor.get('version')!=1:
        raise ValueError('Unsupported image transport version')
    codec=descriptor.get('codec')
    if codec not in ('png','jpeg95') or descriptor.get('color')!='RGB' or descriptor.get('shape')!=[224,224,3]:
        raise ValueError('Malformed image transport descriptor')
    output=dict(observation)
    for name in ('front','wrist'):
        payload=observation[name]
        if not isinstance(payload,bytes) or not 0<len(payload)<=1024*1024:
            raise ValueError('Invalid encoded image payload')
        with Image.open(io.BytesIO(payload)) as image:
            if image.size!=(224,224) or image.format!=('PNG' if codec=='png' else 'JPEG'):
                raise ValueError('Unexpected image dimensions or format')
            output[name]=np.asarray(image.convert('RGB'),dtype=np.uint8).copy()
    output.pop('_image_transport')
    return output


def negotiate_codec(metadata, requested):
    supported=metadata.get('supported_image_codecs',('raw',))
    if requested=='auto':
        # Pixel-identical default; the low-bandwidth launcher explicitly
        # requests JPEG95 so its lossy choice is visible and logged.
        return 'png' if 'png' in supported else 'raw'
    if requested not in supported:
        raise RuntimeError(f'Cloud does not support {requested}; install the image transport patch first')
    return requested
