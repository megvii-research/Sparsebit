import torch


def main(args):
    if torch.cuda.is_available():
        ckpt = torch.load(args.checkpoint)
    else:
        ckpt = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    new_qweight = {}
    for k, v in ckpt["model"].items():
        if "qweight" in k:
            print(k)
            assert v.dtype == torch.int, "weight in checkpoint must be int32!"
            v_list = []
            for i in range(v.shape[0]): # for oc
                v0_8b = (v[i] >> 0*8) & 0xff
                v1_8b = (v[i] >> 1*8) & 0xff
                v2_8b = (v[i] >> 2*8) & 0xff
                v3_8b = (v[i] >> 3*8) & 0xff
                v_8b = torch.cat([v0_8b[None, ...], v1_8b[None, ...], v2_8b[None, ...], v3_8b[None, ...]])
                v_list.append(v_8b)
            new_qweight[k] = torch.cat(v_list).to(torch.int8)

    for k, v in new_qweight.items():
        ckpt["model"][k] = v

    torch.save(ckpt, args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    args = parser.parse_args()

    main(args)
