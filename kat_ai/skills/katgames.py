"""Read-only KatGames progress summarizer; no score/coin mutation methods."""
def summarize_progress(data: dict) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
        raise ValueError("sessions_must_be_a_list")
    sessions = data["sessions"]
    if len(sessions) > 1000:
        raise ValueError("too_many_sessions")
    scores, accuracies, reaction = [], [], []
    for row in sessions:
        if not isinstance(row, dict):
            continue
        score, acc, rt = row.get("score"), row.get("accuracy"), row.get("average_reaction_ms")
        if isinstance(score, (int,float)) and not isinstance(score,bool) and score >= 0: scores.append(float(score))
        if isinstance(acc, (int,float)) and not isinstance(acc,bool) and 0 <= acc <= 1: accuracies.append(float(acc))
        if isinstance(rt, (int,float)) and not isinstance(rt,bool) and 0 < rt < 60000: reaction.append(float(rt))
    return {
        "skill":"KatGames", "mode":"read_only", "sessions":len(sessions),
        "best_score":max(scores) if scores else None,
        "average_score":round(sum(scores)/len(scores),2) if scores else None,
        "accuracy":round(sum(accuracies)/len(accuracies),4) if accuracies else None,
        "average_reaction_ms":round(sum(reaction)/len(reaction),2) if reaction else None,
        "mutations_performed":False
    }
