import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

const PCSelector = ({ selectedX, selectedY, onChange }) => {
  const pcOptions = Array.from({ length: 10 }, (_, i) => i + 1);

  return (
    <div className="space-y-4">
      <div className="grid gap-2">
        <span className="text-sm font-medium">X-axis Principal Component</span>
        <Select
          value={selectedX.toString()}
          onValueChange={(value) => onChange('x', parseInt(value))}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select PC for X-axis" />
          </SelectTrigger>
          <SelectContent>
            {pcOptions.map((pc) => (
              <SelectItem key={pc} value={pc.toString()}>
                PC{pc}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid gap-2">
        <span className="text-sm font-medium">Y-axis Principal Component</span>
        <Select
          value={selectedY.toString()}
          onValueChange={(value) => onChange('y', parseInt(value))}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select PC for Y-axis" />
          </SelectTrigger>
          <SelectContent>
            {pcOptions.map((pc) => (
              <SelectItem key={pc} value={pc.toString()}>
                PC{pc}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
};

export default PCSelector;
