# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately through
[GitHub Security Advisories](https://github.com/ObsidianArch02/fireredasr2s-mlx/security/advisories/new).

Do **not** open a public issue for a vulnerability until a fix has been released.

When reporting, please include:

- The commit hash or release you are using
- A minimal reproduction (script, input file, or command line)
- The observed behavior and the impact you believe it has

You can expect an acknowledgement within a few days. If the issue is confirmed,
a fix will be prepared and released, and credit will be given in the advisory
unless you prefer to remain anonymous.

## Known Security Considerations

### Checkpoint deserialization

`model.pth.tar` files downloaded from upstream are **pickle archives**. The
one-time weight conversion (`fireredasr2_mlx.loader.convert_checkpoint` /
`convert_llm_checkpoint`) must load them with `torch.load(..., weights_only=False)`,
which executes arbitrary code embedded in the file.

- Only convert checkpoints fetched from the official
  [FireRedTeam](https://huggingface.co/FireRedTeam) HuggingFace repositories.
- Never convert a `.pth.tar` file received from an untrusted source, a
  third-party mirror, or a pull request.
- Conversion is a one-time offline step; the inference path
  (`fireredasr2_mlx` at runtime) only loads `safetensors` and never touches
  `torch` or pickle.

### Third-party inputs

- Audio files are processed with `soundfile`/`kaldi-native-fbank`. Only feed
  trusted media; malformed audio parsing vulnerabilities belong to those
  libraries, not this port.
- This project does not ship model weights. Integrity of downloaded weights
  is enforced by fetching from the official HuggingFace repos over HTTPS.

## Scope

This policy covers the code in this repository. Vulnerabilities in the upstream
[FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) reference
implementation should be reported to the upstream project.
