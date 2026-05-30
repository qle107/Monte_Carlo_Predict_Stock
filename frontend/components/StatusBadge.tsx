import { statusLabel } from "@/lib/display";

export default function StatusBadge({
  status,
  tone,
}: {
  status: string;
  tone: string;
}) {
  return (
    <span className="flex items-center gap-1.5 text-[12px] text-muted">
      <span className={"inline-block h-[7px] w-[7px] rounded-full " + tone} />
      {statusLabel(status)}
    </span>
  );
}
