import {
  type GlyphBoxOptions,
  type GlyphBoxPlacements,
  type LabelPlacement,
  type LabelSize,
  type Layout,
  labelObstacles,
  type PlacementStrategy,
  placeGlyphBoxesForRegions,
  placeLabelsForRegions,
  placeSetLabels,
} from "@jolars/eunoia";
import {
  boundingBox,
  type MemberCallout,
  type MemberCalloutLine,
} from "@jolars/eunoia/svg";

export interface MeasuredMemberLine extends LabelSize {
  text: string;
  fontSize: number;
  fontWeight?: string | number;
  fontStyle?: string;
}

interface MemberLayoutOptions {
  layout: Extract<Layout, { mode: "regions" }>;
  labels: Record<string, string[]>;
  sizes: Record<string, LabelSize[]>;
  headerSizes: Record<string, LabelSize>;
  headerLines: Record<string, MeasuredMemberLine[]>;
  missingHeaders: Record<string, MeasuredMemberLine>;
  fontSize: number;
  boxOptions: GlyphBoxOptions;
  strategy: PlacementStrategy;
  setSizes?: Record<string, LabelSize>;
  padding: number;
}

interface MemberLayout {
  boxes: GlyphBoxPlacements;
  placements: Record<string, LabelPlacement>;
  setPlacements?: Record<string, LabelPlacement>;
  callouts: Record<string, MemberCallout>;
}

function rosterBlock(
  headings: readonly MeasuredMemberLine[],
  names: readonly string[],
  sizes: readonly LabelSize[],
  fontSize: number,
): Pick<MemberCallout, "size" | "lines"> {
  const measured: MeasuredMemberLine[] = [
    ...headings,
    ...names.map((text, i) => ({ ...sizes[i], text, fontSize })),
  ];
  const gap = fontSize * 0.2;
  const size = {
    w: Math.max(...measured.map((line) => line.w)),
    h:
      measured.reduce((h, line) => h + line.h, 0) +
      gap * Math.max(0, measured.length - 1),
  };
  let y = -size.h / 2;
  const lines: MemberCalloutLine[] = measured.map((line) => {
    const result = {
      text: line.text,
      y: y + line.h / 2,
      fontSize: line.fontSize,
      fontWeight: line.fontWeight,
      fontStyle: line.fontStyle,
    };
    y += line.h + gap;
    return result;
  });
  return { size, lines };
}

/**
 * Keep each region's roster together, replacing an incomplete packing with a
 * single measured label block. Eviction is permanent for this layout pass,
 * so changing label obstacles cannot make regions alternate between modes.
 */
export function placeMemberLabels(options: MemberLayoutOptions): MemberLayout {
  const { layout, labels, sizes, headerSizes, headerLines, fontSize } = options;
  const geometry = new Set(
    layout.regions.filter((r) => r.pieces.length > 0).map((r) => r.combination),
  );
  const evicted = new Set(
    Object.keys(labels).filter((key) => !geometry.has(key)),
  );
  const blocks: Record<string, Pick<MemberCallout, "size" | "lines">> = {};
  let boxes: GlyphBoxPlacements;
  let placements: Record<string, LabelPlacement>;
  let setPlacements: Record<string, LabelPlacement> | undefined;

  while (true) {
    const labelSizes = { ...headerSizes };
    for (const key of evicted) {
      const headings = geometry.has(key)
        ? (headerLines[key] ?? [])
        : [options.missingHeaders[key]];
      blocks[key] = rosterBlock(headings, labels[key], sizes[key], fontSize);
      if (geometry.has(key)) labelSizes[key] = blocks[key].size;
    }
    placements = placeLabelsForRegions({
      regions: layout.regions,
      container: layout.container,
      sizes: labelSizes,
      strategy: options.strategy,
    });
    const obstacles = labelObstacles({
      placements,
      sizes: labelSizes,
      padding: options.padding,
    });
    setPlacements = options.setSizes
      ? placeSetLabels({
          outlines: layout.shapeOutlines,
          container: layout.container,
          sizes: options.setSizes,
          strategy: { obstacles, precision: options.strategy.precision },
        })
      : undefined;
    if (setPlacements && options.setSizes) {
      obstacles.push(
        ...labelObstacles({
          placements: setPlacements,
          sizes: options.setSizes,
          padding: options.padding,
        }),
      );
    }
    const remaining = Object.fromEntries(
      Object.keys(labels)
        .filter((key) => !evicted.has(key))
        .map((key) => [key, sizes[key]]),
    );
    boxes = placeGlyphBoxesForRegions({
      regions: layout.regions,
      sizes: remaining,
      options: {
        ...options.boxOptions,
        obstacles,
        obstaclePolicy: "strict",
      },
    });
    const failed = Object.keys(remaining).filter(
      (key) => (boxes.boxes[key]?.length ?? 0) !== remaining[key].length,
    );
    if (failed.length === 0) break;
    for (const key of failed) evicted.add(key);
  }

  const callouts: Record<string, MemberCallout> = {};
  const unattached: string[] = [];
  for (const key of evicted) {
    const placement = placements[key];
    if (placement) {
      callouts[key] = { ...blocks[key], placement };
    } else {
      unattached.push(key);
    }
  }
  // Missing or degenerate geometry has no honest leader target. Keep those
  // rosters in a separate column beyond every placed label and the container.
  const bounds = boundingBox(layout, {
    placements,
    labelSizes: {
      ...headerSizes,
      ...Object.fromEntries(
        Object.entries(blocks).map(([k, b]) => [k, b.size]),
      ),
    },
    setLabelPlacements: setPlacements,
    setLabelSizes: options.setSizes,
    memberCallouts: callouts,
  });
  let top = bounds.minY;
  for (const key of unattached) {
    const block = rosterBlock(
      [options.missingHeaders[key]],
      labels[key],
      sizes[key],
      fontSize,
    );
    callouts[key] = {
      ...block,
      placement: {
        anchor: {
          x: bounds.maxX + fontSize + block.size.w / 2,
          y: top + block.size.h / 2,
        },
        kind: "interior",
        leaderWaypoints: [],
      },
    };
    top += block.size.h + fontSize;
  }
  return { boxes, placements, setPlacements, callouts };
}
