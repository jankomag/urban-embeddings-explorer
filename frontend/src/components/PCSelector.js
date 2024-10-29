import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";

const PCSelector = ({ selectedX, selectedY, onChange }) => {
  const pcOptions = [
    { pc: 1, variance: 32.5 },
    { pc: 2, variance: 51.8 },
    { pc: 3, variance: 65.4 },
    { pc: 4, variance: 74.2 },
    { pc: 5, variance: 81.7 },
    { pc: 6, variance: 87.3 }
  ];

  return (
    <div className="flex items-center space-x-4">
      <div className="flex items-center space-x-2">
        <span className="text-sm">X:</span>
        <Select
          value={selectedX.toString()}
          onValueChange={(value) => onChange('x', parseInt(value))}
        >
          <SelectTrigger className="w-[100px]">
            <SelectValue placeholder="PC" />
          </SelectTrigger>
          <SelectContent>
            {pcOptions.map(({ pc, variance }) => (
              <SelectItem key={pc} value={pc.toString()}>
                PC{pc} ({variance.toFixed(1)}%)
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center space-x-2">
        <span className="text-sm">Y:</span>
        <Select
          value={selectedY.toString()}
          onValueChange={(value) => onChange('y', parseInt(value))}
        >
          <SelectTrigger className="w-[100px]">
            <SelectValue placeholder="PC" />
          </SelectTrigger>
          <SelectContent>
            {pcOptions.map(({ pc, variance }) => (
              <SelectItem key={pc} value={pc.toString()}>
                PC{pc} ({variance.toFixed(1)}%)
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
};

export default PCSelector;
