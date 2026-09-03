import ast
import importlib.util
import ntpath
import subprocess
import types
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = (
    REPOSITORY_ROOT
    / "tools"
    / "lam_python_probe"
    / "patches"
    / "lam-one-click-windows.patch"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("<SP>", " "), encoding="utf-8", newline="\n")


def _create_patch_fixture(root: Path) -> None:
    _write(
        root / "app_lam.py",
        """def save_images2video(img_lst, v_pth, fps):
    from moviepy.editor import ImageSequenceClip
    # Ensure all images are in uint8 format
    images = [image.astype(np.uint8) for image in img_lst]
<SP><SP><SP><SP>
    # Create an ImageSequenceClip from the list of images
    clip = ImageSequenceClip(images, fps=fps)
fixture_gap_1 = True
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # prepare motion seq
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
                                            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1,0],
fixture_gap_2 = True
                                        render_c2ws=motion_seq["render_c2ws"].to(device),
                                        render_intrs=motion_seq["render_intrs"].to(device),
                                        render_bg_colors=motion_seq["render_bg_colors"].to(device),
                                        flame_params={k:v.to(device) for k, v in motion_seq["flame_params"].items()})
<SP><SP><SP><SP><SP><SP><SP><SP>
        # save h5 rendering info
        if h5_rendering:
            pass
""",
    )
    _write(
        root / "lam" / "models" / "modeling_lam.py",
        """from collections import defaultdict
import math

import torch
from torch import nn


def rearrange(value, _pattern, **_axes):
    return value


class ModelLAM(nn.Module):
<SP><SP><SP><SP><SP><SP><SP><SP>
    @torch.no_grad()
    def infer_single_view(self, image, source_c2ws, source_intrs, render_c2ws,<SP>
                          render_intrs, render_bg_colors, flame_params):
        # image: [B, N_ref, C_img, H_img, W_img]
        # source_c2ws: [B, N_ref, 4, 4]
        # source_intrs: [B, N_ref, 4, 4]
        # render_c2ws: [B, N_source, 4, 4]
        # render_intrs: [B, N_source, 4, 4]
        # render_bg_colors: [B, N_source, 3]
        # flame_params: Dict, e.g., pose_shape: [B, N_source, 21, 3], betas:[B, 100]
        assert image.shape[0] == render_c2ws.shape[0], "Batch size mismatch for image and render_c2ws"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        assert image.shape[0] == flame_params["betas"].shape[0], "Batch size mismatch for image and flame_params"
        assert image.shape[0] == flame_params["expr"].shape[0], "Batch size mismatch for image and flame_params"
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)
        assert image.shape[0] == 1
        num_views = render_c2ws.shape[1]
        query_points = None
<SP><SP><SP><SP><SP><SP><SP><SP>
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params = self.renderer.get_query_points(flame_params,
                                                                        device=image.device)
        latent_points, image_feats = self.forward_latent_points(image[:, 0], camera=None, query_points=query_points)
        image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))

        gs_model_list, query_points, flame_params, _ = self.renderer.forward_gs(gs_hidden_features=latent_points,
                                                query_points=query_points,
                                                flame_data=flame_params,
                                                additional_features={"image_feats": image_feats, "image": image[:, 0], "image_feats_bchw": image_feats_bchw})

        render_res_list = []
        for view_idx in range(num_views):
            render_res = self.renderer.forward_animate_gs(gs_model_list,<SP>
                                                          query_points,
                                                          self.renderer.get_single_view_smpl_data(flame_params, view_idx),<SP>
                                                          render_c2ws[:, view_idx:view_idx+1],<SP>
                                                          render_intrs[:, view_idx:view_idx+1],<SP>
                                                          render_h,<SP>
                                                          render_w,<SP>
                                                          render_bg_colors[:, view_idx:view_idx+1])
            render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            # print(f"out key:{k}")
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                    out[k] = out[k][0].permute(0, 2, 3, 1)  # [1, Nv, 3, H, W] -> [Nv, 3, H, W] - > [Nv, H, W, 3]<SP>
            else:
                out[k] = v
        out['cano_gs_lst'] = gs_model_list
        return out

""",
    )
    _write(
        root / "lam" / "runners" / "infer" / "lam.py",
        """class LAMInferrer(Inferrer):

    def save_imgs_2_video(self, img_lst, v_pth, fps):
        from moviepy.editor import ImageSequenceClip
        images = [image.astype(np.uint8) for image in img_lst]
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(v_pth, codec='libx264')
        print(f"Video saved successfully at {v_pth}")
fixture_gap_1 = True
        save_img = self.cfg.get("save_img", False)  # False
        rendered_bg = 1.
        ref_bg = 1.
        mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")
        if ref_bg < 1.:
            if "VFHQ_TEST" in image_path:
                mask_path = image_path.replace("/VFHQ_TEST/", "/mask/").replace("/images/", "/mask/").replace(".png", ".jpg")
fixture_gap_2 = True
        # prepare motion seq
        test_sample=self.cfg.get("test_sample", False)
        # test_sample=True
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(motion_seqs_dir, motion_img_dir, save_root=dump_tmp_dir, fps=motion_video_read_fps,
                                            bg_color=rendered_bg, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1,0],
fixture_gap_3 = True
                                               render_c2ws=motion_seq["render_c2ws"].to(device),
                                               render_intrs=motion_seq["render_intrs"].to(device),
                                               render_bg_colors=motion_seq["render_bg_colors"].to(device),
                                               flame_params={k:v.to(device) for k, v in motion_seq["flame_params"].items()})

        print(f"time elapsed: {time.time() - start_time}")
        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        only_pred = rgb
        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)
            pass
fixture_gap_4 = True
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        # images_to_video(rgb, output_path=dump_video_path, fps=render_fps, gradio_codec=False, verbose=True)
        self.save_imgs_2_video(rgb, dump_video_path, render_fps)
        base_vid = motion_seqs_dir.strip('/').split('/')[-1]
        audio_path = os.path.join(motion_seqs_dir, base_vid+".wav")
        dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
        self.add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
""",
    )


def test_patch_applies_to_fixed_baseline_and_preserves_default_call(tmp_path: Path) -> None:
    fixture = tmp_path / "LAM"
    _create_patch_fixture(fixture)

    check = subprocess.run(
        ["git", "apply", "--check", str(PATCH_PATH)],
        cwd=fixture,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    subprocess.run(
        ["git", "apply", str(PATCH_PATH)], cwd=fixture, check=True
    )

    model = (fixture / "lam" / "models" / "modeling_lam.py").read_text(
        encoding="utf-8"
    )
    runner = (fixture / "lam" / "runners" / "infer" / "lam.py").read_text(
        encoding="utf-8"
    )
    app = (fixture / "app_lam.py").read_text(encoding="utf-8")
    assert "render_output_keys=None, render_output_device=None" in model
    assert 'render_output_device="cpu"' in runner
    assert "clamp_(0, 1.0).mul_(255)" in runner
    assert "os.path.normpath(image_path)" in app


def test_patched_model_preserves_default_and_streams_selected_outputs(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    fixture = tmp_path / "LAM"
    _create_patch_fixture(fixture)
    subprocess.run(["git", "apply", str(PATCH_PATH)], cwd=fixture, check=True)

    module_path = fixture / "lam" / "models" / "modeling_lam.py"
    spec = importlib.util.spec_from_file_location("patched_modeling_lam", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeRenderer:
        def __init__(self) -> None:
            self.view_index = 0

        def forward_gs(self, **kwargs):
            return ["canonical"], None, kwargs["flame_data"], None

        def get_single_view_smpl_data(self, _flame_params, _view_index):
            return {}

        def forward_animate_gs(self, *_args):
            view_index = self.view_index
            self.view_index += 1
            return {
                "comp_rgb": torch.full((1, 1, 3, 2, 2), view_index + 1.0),
                "aux": torch.full((1, 1, 1), view_index + 10.0),
                "3dgs": f"gs-{view_index}",
            }

    def create_model():
        model = module.ModelLAM()
        model.latent_query_points_type = "fixed"
        model.renderer = FakeRenderer()
        model.forward_latent_points = lambda *_args, **_kwargs: (
            torch.zeros((1, 1, 1)),
            torch.zeros((1, 1, 1)),
        )
        return model

    image = torch.zeros((1, 1, 3, 1, 1))
    cameras = torch.zeros((1, 2, 4, 4))
    intrinsics = torch.zeros((1, 2, 4, 4))
    intrinsics[:, :, 0, 2] = 1
    intrinsics[:, :, 1, 2] = 1
    backgrounds = torch.zeros((1, 2, 3))
    flame_params = {
        "betas": torch.zeros((1, 1)),
        "expr": torch.zeros((1, 2, 1)),
    }

    default_output = create_model().infer_single_view(
        image,
        cameras[:, :1],
        intrinsics[:, :1],
        cameras,
        intrinsics,
        backgrounds,
        flame_params,
    )
    streamed_output = create_model().infer_single_view(
        image,
        cameras[:, :1],
        intrinsics[:, :1],
        cameras,
        intrinsics,
        backgrounds,
        flame_params,
        render_output_keys=("comp_rgb", "3dgs"),
        render_output_device="cpu",
    )

    assert default_output["comp_rgb"].shape == (2, 2, 2, 3)
    assert torch.equal(default_output["comp_rgb"], streamed_output["comp_rgb"])
    assert streamed_output["comp_rgb"].device.type == "cpu"
    assert streamed_output["3dgs"] == ["gs-0", "gs-1"]
    assert "aux" in default_output
    assert "aux" not in streamed_output


def test_patched_windows_path_expression_uses_native_separators(tmp_path: Path) -> None:
    fixture = tmp_path / "LAM"
    _create_patch_fixture(fixture)
    subprocess.run(["git", "apply", str(PATCH_PATH)], cwd=fixture, check=True)
    app_lines = (fixture / "app_lam.py").read_text(encoding="utf-8").splitlines()
    expressions = {
        line.strip().split(" = ", 1)[0]: line.strip().split(" = ", 1)[1]
        for line in app_lines
        if line.strip().startswith(("src = ", "driven = "))
    }
    path_module = types.SimpleNamespace(path=ntpath)
    source = eval(
        compile(ast.parse(expressions["src"], mode="eval"), "<src>", "eval"),
        {"os": path_module, "image_path": r"C:\\work\\assets\\sample_input\\status.png"},
    )
    driven = eval(
        compile(ast.parse(expressions["driven"], mode="eval"), "<driven>", "eval"),
        {"os": path_module, "motion_seqs_dir": r"C:\\work\\motions\\Look_In_My_Eyes"},
    )

    assert source == "assets"
    assert driven == "Look_In_My_Eyes"


def test_readme_patch_command_is_repository_relative() -> None:
    readme = (REPOSITORY_ROOT / "tools" / "lam_python_probe" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "C:\\projects\\cpp\\KFCore" not in readme
    assert "$repoRoot = git rev-parse --show-toplevel" in readme
    assert "--unidiff-zero" not in readme
