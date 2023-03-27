import torch, av, os, re, math, gc
from typing import Union, Optional, Dict, Tuple, Any, List
from fractions import Fraction
import numpy as np

from torchvision.utils import _log_api_usage_once

# PyAV has some reference cycles
_CALLED_TIMES = 0
_GC_COLLECTION_INTERVAL = 10

def _read_from_stream(
    container: "av.container.Container",
    start_offset: float,
    end_offset: float,
    pts_unit: str,
    stream: "av.stream.Stream",
    stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
) -> List["av.frame.Frame"]:
    global _CALLED_TIMES, _GC_COLLECTION_INTERVAL
    _CALLED_TIMES += 1
    if _CALLED_TIMES % _GC_COLLECTION_INTERVAL == _GC_COLLECTION_INTERVAL - 1:
        gc.collect()

    assert pts_unit == "pts"

    frames = {}
    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError:
        # TODO add some warnings in this case
        # print("Corrupted file?", container.name)
        return []
    buffer_count = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames[frame.pts] = frame
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError:
        # TODO add a warning
        pass
    # ensure that the results are sorted wrt the pts
    result = [frames[i] for i in sorted(frames) if start_offset <= frames[i].pts <= end_offset]
    if len(frames) > 0 and start_offset > 0 and start_offset not in frames:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        preceding_frames = [i for i in frames if i < start_offset]
        if len(preceding_frames) > 0:
            first_frame_pts = max(preceding_frames)
            result.insert(0, frames[first_frame_pts])
    return result

def read_video_without_audio(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(read_video_without_audio)

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(
            f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
        )

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
            )

    except av.AVError:
        # TODO raise a warning?
        pass
        
    vframes = torch.as_tensor(np.stack([frame.to_rgb().to_ndarray() for frame in video_frames]))
    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes