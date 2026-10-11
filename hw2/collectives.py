import torch
import torch.distributed as dist


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    previous = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        recv_index = (rank - step - 2) % world_size
        received = torch.empty_like(chunks[recv_index])
        send = dist.isend(chunks[send_index], dst=next_rank)
        recv = dist.irecv(received, src=previous)
        send.wait()
        recv.wait()
        chunks[recv_index].add_(received)
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    previous = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        send = dist.isend(chunks[send_index], dst=next_rank)
        recv = dist.irecv(chunks[recv_index], src=previous)
        send.wait()
        recv.wait()
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    reduced_shard = ring_reduce_scatter(tensor)
    return ring_all_gather(reduced_shard)
